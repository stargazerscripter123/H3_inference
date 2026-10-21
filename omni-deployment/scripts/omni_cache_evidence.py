"""Owned-worker, numeric-only cache request evidence. No raw log text escapes."""
import hashlib
import json
from pathlib import Path
import re

MARKER = 'H3_CONDITIONING_CACHE_EVENT '
PLACEMENTS = {'park', 'ensure_encoder', 'release_encoder', 'load_denoise', 'restore'}
KEYS = {'schema_version', 'rank', 'ranks', 'request_ordinal', 'enabled', 'diagnostic', 'decision',
        'hits', 'placement_calls', 'dit_forward_count', 'event', 'completed'}


def parse_event(line):
    if MARKER not in line:
        return None
    payload = line.split(MARKER, 1)[1].lstrip()
    value, end = json.JSONDecoder().raw_decode(payload)
    tail = payload[end:]
    # Ray adds ANSI resets only. Unknown trailing text cannot be evidence.
    tail = re.sub(r'\x1b\[[0-9;]*m', '', tail).strip()
    if tail or not isinstance(value, dict) or set(value) != KEYS:
        raise ValueError('cache_event_schema')
    for key in ('schema_version', 'rank', 'ranks', 'request_ordinal'):
        if type(value[key]) is not int:
            raise ValueError('cache_event_numeric_type')
    if (value['schema_version'] != 1 or value['rank'] not in (0, 1) or value['ranks'] != 2
            or value['request_ordinal'] not in (1, 2, 3, 4)):
        raise ValueError('cache_event_rank_or_request')
    if type(value['enabled']) is not bool or type(value['diagnostic']) is not bool:
        raise ValueError('cache_event_boolean_type')
    if (value['decision'] not in {'pending', 'off', 'miss', 'hit', 'error'} or value['event'] not in {'begin', 'end'}
            or value['completed'] is not None and type(value['completed']) is not bool):
        raise ValueError('cache_event_enum')
    calls = value['placement_calls']
    if (not isinstance(calls, dict) or set(calls) != PLACEMENTS
            or any(type(v) is not int or v not in (0, 1) for v in calls.values())):
        raise ValueError('cache_event_placement_count')
    if value['hits'] is not None and (type(value['hits']) is not int or not 0 <= value['hits'] <= 2):
        raise ValueError('cache_event_hit_count')
    count = value['dit_forward_count']
    if count is not None and (type(count) is not int or not 0 <= count <= 8):
        raise ValueError('cache_event_forward_count')
    return value


def validate_events(by_rank, *, enabled, diagnostic):
    if (type(enabled) is not bool or type(diagnostic) is not bool or set(by_rank) != {0, 1}
            or any(type(rank) is not int for rank in by_rank)):
        raise ValueError('cache_events_two_ranks_required')
    ends = {}
    for rank, rows in by_rank.items():
        if len(rows) != 8:
            raise ValueError('cache_events_four_begin_end_pairs_required')
        ends[rank] = []
        for i in range(4):
            begin, end = rows[2*i:2*i+2]
            for row in (begin, end):
                if (row['rank'] != rank or row['request_ordinal'] != i+1 or row['enabled'] is not enabled
                        or row['diagnostic'] is not diagnostic):
                    raise ValueError('cache_events_rank_order_or_mode_mismatch')
            if (begin['event'] != 'begin' or begin['completed'] is not None
                    or begin['decision'] != ('pending' if enabled else 'off') or begin['hits'] is not None
                    or any(begin['placement_calls'].values())
                    or begin['dit_forward_count'] != (0 if diagnostic else None)):
                raise ValueError('cache_events_begin_invalid')
            decision = 'off' if not enabled else 'miss' if i == 0 else 'hit'
            if (end['event'] != 'end' or end['completed'] is not True or end['decision'] != decision
                    or end['hits'] != (None if not enabled else 0 if i == 0 else 2)
                    or set(end['placement_calls'].values()) != ({0} if decision == 'hit' else {1})
                    or end['dit_forward_count'] != (8 if diagnostic else None)):
                raise ValueError('cache_events_expected_execution_unverified')
            ends[rank].append(end)
    return ends


def collect_cache_events(service, service_log, reader, *, enabled, diagnostic, identity_verified,
                         tmp_root=Path('/tmp')):
    result = {'validated': False, 'instrumented': diagnostic, 'rank_rows': {}, 'reason': None,
              'binding': 'owned_ray_pid_plus_explicit_rank_and_serial_request_ordinal',
              'placement_scope': 'method invocation counts; not measured PCIe bytes', 'worker_log_sha256': []}
    try:
        if not identity_verified:
            raise ValueError('cache_launch_identity_unverified')
        token, uid = service.get('owner_token'), service.get('uid')
        if (not isinstance(token, str) or not re.fullmatch('[0-9a-f]{32}', token)
                or type(uid) is not int or uid < 0):
            raise ValueError('cache_private_ray_identity_unverified')
        pids = reader.worker_pids.get(service_log, set())
        if not pids:
            raise ValueError('cache_worker_pid_link_missing')
        base = tmp_root.resolve() / f'h3-omni-{uid}-{token[:8]}'
        if base != base.resolve() or base.is_symlink():
            raise ValueError('cache_ray_directory_alias')
        by_rank, used_pids = {}, set()
        for path in sorted(base.glob('session_*/logs/worker-*')):
            match = re.fullmatch(r'worker-[0-9a-f]+-[0-9a-f]+-(\d+)\.(out|err)', path.name)
            if not match or int(match[1]) not in pids:
                continue
            if path != path.resolve() or not path.is_file():
                raise ValueError('cache_worker_log_alias')
            entries, sha = [], hashlib.sha256()
            with path.open('rb') as stream:
                for raw in stream:
                    sha.update(raw)
                    if MARKER.encode() in raw:
                        entries.append(parse_event(raw.decode('utf8')))
            if not entries:
                continue
            ranks = {entry['rank'] for entry in entries}
            if len(ranks) != 1:
                raise ValueError('cache_worker_rank_ambiguous')
            rank = ranks.pop()
            pid = int(match[1])
            if rank in by_rank or pid in used_pids:
                raise ValueError('cache_duplicate_worker_evidence')
            by_rank[rank] = entries
            used_pids.add(pid)
            result['worker_log_sha256'].append(sha.hexdigest())
        result['rank_rows'] = validate_events(by_rank, enabled=enabled, diagnostic=diagnostic)
        result['validated'] = True
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
        # Error strings here are our constants; malformed JSON library messages
        # can contain private data, so expose only a closed set of our codes.
        message = str(error)
        result['reason'] = message if re.fullmatch('cache_[a-z_]+', message) else 'cache_event_read_or_parse_failed'
    return result


def validate_diagnostic_evidence(value, *, variant, protocol_sha256, plan_sha256, decoder_sha256, source_stack):
    """Recheck sanitized evidence against this run; labels/validated alone never suffice."""
    def need(ok):
        if not ok: raise ValueError('cache_diagnostic_contract_or_execution_mismatch')
    meta = value.get('optimization', {})
    jobs = value.get('jobs', [])
    need(value.get('collector') == 'collect_cache_eval' and value.get('run_status') == 'completed'
         and value.get('protocol_validated') is True and value.get('protocol_sha256') == protocol_sha256
         and variant in {'cache-off', 'cache-on'} and meta.get('variant') == variant
         and meta.get('source_stack_contract_sha256') == source_stack['stack_contract_sha256']
         and meta.get('decoder_sha256') == decoder_sha256 and meta.get('sp_profile') is True
         and meta.get('formal_timing_eligible') is False and meta.get('selected_cases') == ['case-01']
         and len(jobs) == 1)
    job=jobs[0]; recipe=job.get('runtime_recipe', {}); cache=recipe.get('cache_evidence', {})
    need(job.get('validation_reasons') == ['diagnostic_run_excluded_from_formal_timing'])
    need(job.get('case_id') == 'case-01' and job.get('policy_id') == 'cap480' and job.get('status') == 'passed'
         and job.get('plan_sha256') == plan_sha256 and job.get('service_final_state') == 'STOPPED'
         and job.get('cleanup_error_count') == 0 and recipe.get('validated') is True
         and recipe.get('source_stack') == source_stack and cache.get('validated') is True
         and cache.get('instrumented') is True)
    actual=recipe.get('actual', {})
    need(all(actual.get(k) == v and type(actual.get(k)) is type(v) for k,v in {
        'decoder_sha256':decoder_sha256, 'sp_profile':True,'stage_logging':True,'nvfp4_mm_backend':'auto',
        'nvfp4_mm_row_split':'off','vae_offload':True,'expandable_segments':True,
        'resize_policy':'cap','max_short_edge':480,'conditioning_cache_fastpath':variant=='cache-on'}.items()))
    requests=job.get('requests', [])
    need(len(requests)==4)
    for i,row in enumerate(requests):
        need(type(row.get('seed')) is int and row['seed']==(9000,1234,5678,9012)[i]
             and row.get('phase')==('warmup' if i==0 else 'measured')
             and row.get('outcome') in {'succeeded','succeeded_recovered_from_manifest'}
             and row.get('media_validated_by_client') is True and row.get('included') is False
             and row.get('exclusion_reasons') == ['diagnostic_run_excluded_from_formal_timing'])
    rows=cache.get('rank_rows', {})
    need(isinstance(rows,dict) and set(rows)=={'0','1'})
    for rank in (0,1):
        need(isinstance(rows[str(rank)],list) and len(rows[str(rank)])==4)
        for i,row in enumerate(rows[str(rank)]):
            row=parse_event(MARKER+json.dumps(row,allow_nan=False))
            hit=variant=='cache-on' and i>0
            need(row['rank']==rank and row['request_ordinal']==i+1 and row['enabled'] is (variant=='cache-on')
                 and row['diagnostic'] is True and row['event']=='end' and row['completed'] is True
                 and row['decision']==('hit' if hit else 'miss' if variant=='cache-on' else 'off')
                 and row['hits']==(2 if hit else 0 if variant=='cache-on' else None)
                 and row['dit_forward_count']==8 and set(row['placement_calls'].values())==({0} if hit else {1}))
    return True
