# =============================================================================
# File        : twindrive/world/road_graph.py
# Project     : TwinDrive — Lane-Free Autonomous Driving on a Digital Twin
#               of an Indian Campus (SRM Town / cm2, CARLA 0.9.16)
# Phase       : 0 — Map readiness
# =============================================================================
"""
Description
-----------
Turns the CARLA road network into a directed graph and finds which lanes are
"routable": lanes from which you can drive to every other routable lane AND
come back. Those lanes form the biggest strongly connected component (SCC).

Why it matters: on cm2 the outer highway and a few stubs end in dead ends.
If a route goal is placed there, the car can drive in but never out. All
route goals in TwinDrive are restricted to the routable set.

Graph definition
    node  = one lane segment, identified by (road_id, section_id, lane_id)
    edge  = A -> B if driving off the end of lane A puts you on lane B

Public functions
    lane_key(wp)                      -> (road, section, lane)
    build_lane_graph(cmap)            -> (graph, dead_end_waypoints)
    strongly_connected_components(g)  -> list of node lists
    routable_lane_keys(cmap)          -> set of keys in the biggest SCC
    is_routable(wp, routable)         -> bool, with a fallback for lanes whose
                                         section id is not a graph node
"""


def lane_key(wp):
    """Identify one lane segment: (road_id, section_id, lane_id)."""
    return (wp.road_id, wp.section_id, wp.lane_id)


def build_lane_graph(cmap, step=1.0):
    """Directed lane graph from the map topology.

    Returns
        graph     : dict key -> set(successor keys)
        dead_ends : list of waypoints where a lane ends with no successor
    """
    graph, dead_ends = {}, []
    for entry_wp, exit_wp in cmap.get_topology():
        k = lane_key(entry_wp)
        graph.setdefault(k, set())
        successors = exit_wp.next(step)
        if not successors:
            dead_ends.append(exit_wp)
        for s in successors:
            sk = lane_key(s)
            graph.setdefault(sk, set())
            if sk != k:
                graph[k].add(sk)
    return graph, dead_ends


def strongly_connected_components(graph):
    """Iterative Kosaraju's algorithm (no recursion limit problems).

    Pass 1: DFS on the graph, record nodes in order of finishing.
    Pass 2: DFS on the REVERSED graph, in reverse finishing order.
            Each pass-2 tree is one SCC.
    """
    nodes = set(graph)
    for succs in graph.values():
        nodes |= succs

    visited, order = set(), []
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, iter(graph.get(start, ())))]
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                order.append(node)
            elif nxt not in visited:
                visited.add(nxt)
                stack.append((nxt, iter(graph.get(nxt, ()))))

    reverse = {n: set() for n in nodes}
    for n, succs in graph.items():
        for s in succs:
            reverse[s].add(n)

    assigned, components = set(), []
    for start in reversed(order):
        if start in assigned:
            continue
        assigned.add(start)
        comp, stack = [], [start]
        while stack:
            n = stack.pop()
            comp.append(n)
            for p in reverse[n]:
                if p not in assigned:
                    assigned.add(p)
                    stack.append(p)
        components.append(comp)
    return components


def routable_lane_keys(cmap):
    """Lane keys in the biggest SCC: the part of campus you can route within."""
    graph, _ = build_lane_graph(cmap)
    comps = strongly_connected_components(graph)
    return set(max(comps, key=len)) if comps else set()


def is_routable(wp, routable, _cache={}):
    """True if waypoint wp lies on a routable lane.

    Exact match on (road, section, lane) first. If that key is not a graph
    node (a lane split into extra sections mid-road), fall back to matching
    (road, lane) against the routable set.
    """
    if lane_key(wp) in routable:
        return True
    rid = id(routable)
    if rid not in _cache:
        _cache.clear()
        _cache[rid] = {(r, l) for (r, _s, l) in routable}
    return (wp.road_id, wp.lane_id) in _cache[rid]
