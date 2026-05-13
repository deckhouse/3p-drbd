// SPDX-License-Identifier: GPL-2.0-only
/*
 * drbd_admin_lock.c -- cluster-wide administrative lock for DRBD resources.
 *
 * The admin_lock is a single, per-resource boolean that, while held,
 * causes drbd_adm_prepare() to reject any administrative netlink
 * command not on a small IO-orchestration whitelist (suspend-io,
 * resume-io, track-bitmap, flush-bitmap, new-current-uuid, plus the
 * lock/unlock/force-unlock commands themselves). It is acquired and
 * released atomically across all reachable peers via TWOPC_ADMIN_LOCK,
 * and reconciled on reconnect via the P_ADMIN_LOCK_STATE handshake
 * exchange.
 *
 * The lock has no kernel-level timeout. Lifecycle is owned by the
 * userspace caller (typically a Kubernetes controller managing
 * ReplicatedVolumeSnapshot / ReplicatedVolume clone operations); a
 * dedicated DRBD_ADM_FORCE_UNLOCK exists as the operator escape hatch
 * for the case where the holder node has died permanently.
 *
 * Wire format and packet flow:
 *   - DRBD_FF_ADMIN_LOCK is the negotiated feature flag.
 *   - P_TWOPC_PREP_LOCK is the prepare-phase packet (commit reuses
 *     P_TWOPC_COMMIT / P_TWOPC_ABORT).
 *   - TWOPC_ADMIN_LOCK_OP_LOCK / OP_UNLOCK in twopc_request flags
 *     distinguish acquire from release.
 *   - P_ADMIN_LOCK_STATE carries the local view (held, holder,
 *     generation, seq) during state-handshake.
 *
 * See docs/dev/drbd-kernel-admin-lock-design.md for the full design.
 */

#include <linux/jiffies.h>
#include <linux/sched.h>
#include <linux/wait.h>

#include "drbd_int.h"
#include "drbd_protocol.h"
#include "linux/drbd.h"
#include "linux/drbd_genl_api.h"
#include "linux/drbd_limits.h"

/*
 * Feature support: every connected peer must advertise
 * DRBD_FF_ADMIN_LOCK in its agreed_features.
 *
 * Modeled on drbd_support_2pc_resize() in drbd_receiver.c.
 */
enum drbd_state_rv drbd_support_admin_lock(struct drbd_resource *resource)
{
	struct drbd_connection *connection;
	enum drbd_state_rv rv = SS_SUCCESS;

	rcu_read_lock();
	for_each_connection_rcu(connection, resource) {
		if (connection->cstate[NOW] != C_CONNECTED)
			continue;
		if (!(connection->agreed_features & DRBD_FF_ADMIN_LOCK)) {
			rv = SS_NOT_SUPPORTED;
			break;
		}
	}
	rcu_read_unlock();

	return rv;
}

/*
 * Whitelist of admin commands that bypass the admin_lock gate.
 * Centralized so the IO-orchestration policy is auditable in one place.
 *
 * IMPORTANT: keep this list MINIMAL. Anything added here is reachable
 * even on a locked resource and may interfere with snapshot/clone
 * operations in flight. The current set is exactly what RVS / RV-clone
 * need to drive their per-step state machines.
 */
static bool admin_cmd_is_lock_bypass(u8 cmd)
{
	switch (cmd) {
	/* The lock-management commands themselves. Without these on the
	 * whitelist, a held lock would be impossible to release through
	 * the normal path. */
	case DRBD_ADM_LOCK:
	case DRBD_ADM_UNLOCK:
	case DRBD_ADM_FORCE_UNLOCK:
	/* IO orchestration for snapshot / clone flows. */
	case DRBD_ADM_SUSPEND_IO:
	case DRBD_ADM_RESUME_IO:
	case DRBD_ADM_NEW_C_UUID:
	case DRBD_ADM_TRACK_BITMAP:
	case DRBD_ADM_FLUSH_BITMAP:
	/* Get-* read-only queries. They actually use .dumpit and never
	 * reach drbd_adm_prepare(), so listing them here is purely
	 * defensive in case the netlink layout changes. */
	case DRBD_ADM_GET_TIMEOUT_TYPE:
	case DRBD_ADM_GET_RESOURCES:
	case DRBD_ADM_GET_DEVICES:
	case DRBD_ADM_GET_CONNECTIONS:
	case DRBD_ADM_GET_PEER_DEVICES:
	case DRBD_ADM_GET_PATHS:
	case DRBD_ADM_GET_INITIAL_STATE:
		return true;
	default:
		return false;
	}
}

bool drbd_admin_lock_blocks(struct drbd_resource *resource, u8 cmd)
{
	if (!READ_ONCE(resource->admin_lock.held))
		return false;
	return !admin_cmd_is_lock_bypass(cmd);
}

/*
 * Returns true if every directly-connected peer_device on `resource`
 * has repl_state == L_ESTABLISHED. Disconnected peers are skipped
 * (they cannot interfere with our local view of "no resync running").
 */
static bool all_peers_locally_established(struct drbd_resource *resource)
{
	struct drbd_connection *connection;
	struct drbd_peer_device *peer_device;
	bool established = true;
	int vnr;

	rcu_read_lock();
	for_each_connection_rcu(connection, resource) {
		if (connection->cstate[NOW] != C_CONNECTED)
			continue;
		idr_for_each_entry(&connection->peer_devices, peer_device, vnr) {
			if (peer_device->repl_state[NOW] != L_ESTABLISHED) {
				established = false;
				goto out;
			}
		}
	}
out:
	rcu_read_unlock();
	return established;
}

enum drbd_state_rv drbd_admin_lock_wait_for_local_drain(struct drbd_resource *resource)
{
	long timeout_jiffies;
	long ret;

	if (all_peers_locally_established(resource))
		return SS_SUCCESS;

	drbd_info(resource,
		"admin_lock: waiting up to %us for local resync drain before twopc\n",
		resource->res_opts.admin_lock_wait_timeout);

	timeout_jiffies = msecs_to_jiffies(
		resource->res_opts.admin_lock_wait_timeout * 1000U);

	ret = wait_event_interruptible_timeout(
		resource->state_wait,
		all_peers_locally_established(resource),
		timeout_jiffies);

	if (ret < 0)
		return SS_UNKNOWN_ERROR; /* signal */
	if (ret == 0)
		return SS_TIMEOUT;
	return SS_SUCCESS;
}

/*
 * Peer-side prepare: decide whether to vote YES or NO for an incoming
 * TWOPC_ADMIN_LOCK. Called from process_twopc() in drbd_receiver.c
 * with the staged twopc.admin_lock state already populated by the
 * payload-parsing switch.
 *
 * The peer does NOT wait for any local drain: the coordinator already
 * waited under its own adm_mutex, and this packet is processed in the
 * receiver thread where blocking would stall network IO.
 */
enum drbd_state_rv drbd_admin_lock_twopc_prepare_peer(struct drbd_resource *resource,
						      struct twopc_admin_lock *al)
{
	bool held;
	int holder;
	u32 gen;

	read_lock_irq(&resource->state_rwlock);
	held = resource->admin_lock.held;
	holder = resource->admin_lock.holder_node_id;
	gen = resource->admin_lock.generation_tid;
	read_unlock_irq(&resource->state_rwlock);

	if (al->is_lock) {
		/* Acquire: refuse if anybody else already holds it.
		 * Idempotent re-acquire by the same holder/generation is
		 * accepted. */
		if (held && (holder != al->holder_node_id ||
			     gen != al->generation_tid)) {
			drbd_info(resource,
				"TWOPC_ADMIN_LOCK acquire refused: locally held by node %d gen %u, "
				"requested by node %d gen %u\n",
				holder, gen, al->holder_node_id, al->generation_tid);
			return SS_CW_FAILED_BY_PEER;
		}
		/* Refuse if any local peer connection is not Established
		 * (background resync running). The coordinator will retry
		 * with backoff after waiting in drbd_admin_lock_wait_for_local_drain
		 * on its own side. */
		if (!all_peers_locally_established(resource)) {
			drbd_info(resource,
				"TWOPC_ADMIN_LOCK acquire deferred: local resync in progress\n");
			return SS_CW_FAILED_BY_PEER;
		}
		return SS_SUCCESS;
	}

	/* Release: idempotent if already !held. */
	if (!held)
		return SS_SUCCESS;
	/* Refuse cross-holder release; the holder identity is enforced
	 * end-to-end so a stale unlock initiator does not trample over a
	 * fresh holder. */
	if (holder != al->holder_node_id || gen != al->generation_tid) {
		drbd_info(resource,
			"TWOPC_ADMIN_LOCK release refused: held by node %d gen %u, "
			"unlock initiated by node %d gen %u\n",
			holder, gen, al->holder_node_id, al->generation_tid);
		return SS_CW_FAILED_BY_PEER;
	}
	return SS_SUCCESS;
}

/*
 * Peer-side commit: write the agreed-upon admin_lock state to the
 * resource and bump seq. Called from process_twopc() in drbd_receiver.c.
 */
void drbd_admin_lock_twopc_commit_peer(struct drbd_resource *resource,
				       struct twopc_admin_lock *al)
{
	write_lock_irq(&resource->state_rwlock);
	if (al->is_lock) {
		resource->admin_lock.held = true;
		resource->admin_lock.holder_node_id = al->holder_node_id;
		resource->admin_lock.generation_tid = al->generation_tid;
	} else {
		resource->admin_lock.held = false;
		resource->admin_lock.holder_node_id = -1;
		resource->admin_lock.generation_tid = 0;
	}
	resource->admin_lock.seq++;
	write_unlock_irq(&resource->state_rwlock);

	drbd_info(resource,
		"admin_lock %s committed (holder=%d gen=%u seq=%llu)\n",
		al->is_lock ? "ACQUIRE" : "RELEASE",
		al->holder_node_id, al->generation_tid,
		(unsigned long long)resource->admin_lock.seq);
}

/*
 * Send the local admin_lock view to a single peer. Called once per
 * connection from conn_connect2() during state-handshake, and from
 * drbd_admin_lock_broadcast_state() after a force-unlock.
 *
 * No-op (with debug log) if the peer does not advertise
 * DRBD_FF_ADMIN_LOCK; an old peer without the feature would not know
 * how to parse the packet.
 */
int drbd_send_admin_lock_state(struct drbd_connection *connection)
{
	struct drbd_resource *resource = connection->resource;
	struct p_admin_lock_state *p;
	bool held;
	int holder;
	u32 gen;
	u64 seq;

	if (!(connection->agreed_features & DRBD_FF_ADMIN_LOCK))
		return 0;

	read_lock_irq(&resource->state_rwlock);
	held = resource->admin_lock.held;
	holder = resource->admin_lock.holder_node_id;
	gen = resource->admin_lock.generation_tid;
	seq = resource->admin_lock.seq;
	read_unlock_irq(&resource->state_rwlock);

	p = conn_prepare_command(connection, sizeof(*p), DATA_STREAM);
	if (!p)
		return -EIO;

	p->seq = cpu_to_be64(seq);
	p->generation_tid = cpu_to_be32(gen);
	p->holder_node_id = (int8_t)holder;
	p->held = held ? 1 : 0;
	p->_pad[0] = 0;
	p->_pad[1] = 0;

	return send_command(connection, -1, P_ADMIN_LOCK_STATE, DATA_STREAM);
}

/*
 * Merge an incoming peer view with the local admin_lock state.
 *
 * Tie-break: peer_seq > local_seq → adopt peer view.
 *            peer_seq < local_seq → ignore (we are newer).
 *            peer_seq == local_seq → views must agree on (held, holder,
 *            generation); any disagreement is logged as a warning and
 *            the local view is kept (caller cannot do better).
 *
 * This is the only point where admin_lock can change without a twopc
 * commit; the bump-on-adopt rule (we copy peer_seq, we don't bump it)
 * preserves seq monotonicity end-to-end.
 */
void drbd_admin_lock_apply_peer_view(struct drbd_resource *resource,
				     bool peer_held,
				     int peer_holder_node_id,
				     u32 peer_generation_tid,
				     u64 peer_seq)
{
	bool adopt = false;
	bool disagree = false;

	write_lock_irq(&resource->state_rwlock);

	if (peer_seq > resource->admin_lock.seq) {
		adopt = true;
	} else if (peer_seq == resource->admin_lock.seq) {
		if (peer_held != resource->admin_lock.held ||
		    (peer_held &&
		     (peer_holder_node_id != resource->admin_lock.holder_node_id ||
		      peer_generation_tid != resource->admin_lock.generation_tid)))
			disagree = true;
	}

	if (adopt) {
		resource->admin_lock.held = peer_held;
		resource->admin_lock.holder_node_id =
			peer_held ? peer_holder_node_id : -1;
		resource->admin_lock.generation_tid =
			peer_held ? peer_generation_tid : 0;
		resource->admin_lock.seq = peer_seq;
	}

	write_unlock_irq(&resource->state_rwlock);

	if (adopt) {
		drbd_info(resource,
			"admin_lock: adopted peer view (held=%d holder=%d gen=%u seq=%llu)\n",
			peer_held, peer_holder_node_id, peer_generation_tid,
			(unsigned long long)peer_seq);
	} else if (disagree) {
		drbd_warn(resource,
			"admin_lock: peer view at equal seq disagrees with local "
			"(local held=%d holder=%d gen=%u; peer held=%d holder=%d gen=%u; seq=%llu) "
			"- keeping local; investigate split-brain admin_lock\n",
			resource->admin_lock.held,
			resource->admin_lock.holder_node_id,
			resource->admin_lock.generation_tid,
			peer_held, peer_holder_node_id, peer_generation_tid,
			(unsigned long long)peer_seq);
	}
}

/*
 * Push the local admin_lock view to every connected peer that
 * advertises DRBD_FF_ADMIN_LOCK. Used by DRBD_ADM_FORCE_UNLOCK to
 * propagate a unilateral release without going through twopc.
 *
 * Best-effort: per-peer send errors are logged but not surfaced. A
 * peer that misses this packet will adopt the unlocked view via the
 * regular handshake exchange on (re)connect.
 */
void drbd_admin_lock_broadcast_state(struct drbd_resource *resource)
{
	struct drbd_connection *connection;
	u64 im;
	int err;

	for_each_connection_ref(connection, im, resource) {
		if (connection->cstate[NOW] != C_CONNECTED)
			continue;
		if (!(connection->agreed_features & DRBD_FF_ADMIN_LOCK))
			continue;
		err = drbd_send_admin_lock_state(connection);
		if (err)
			drbd_warn(connection,
				"admin_lock broadcast: send failed (err=%d); "
				"peer will reconcile on next handshake\n", err);
	}
}
