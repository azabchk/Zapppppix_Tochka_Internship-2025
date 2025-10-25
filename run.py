import sys
from heapq import heappush, heappop
from typing import List, Tuple, Iterable, Dict

# ---------------- Constants ----------------
COSTS = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}
ROOM_POS = (2, 4, 6, 8)
FORBIDDEN_STOPS = set(ROOM_POS)
TARGET = ('A', 'B', 'C', 'D')
TYPE_TO_ROOM = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
HALL_LEN = 11
INF = float('inf')

State = Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...]]  # (hallway, rooms-as-stacks)

# ---------------- Parsing ----------------
def parse(lines: List[str]) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, ...], ...], int]:
    """
    Parse input lines into initial state.
    
    Returns:
        (hallway, rooms, depth) where:
        - hallway: tuple of 11 positions
        - rooms: tuple of 4 room stacks (bottom to top)
        - depth: room depth (2 or 4)
    """
    lines = [ln.rstrip('\n') for ln in lines if ln.strip() != ""]
    
    # Hallway: line that starts/ends with '#' and has exactly 11 inner cells
    hallway = tuple('.' for _ in range(HALL_LEN))
    for ln in lines:
        if ln.startswith('#') and ln.endswith('#') and len(ln[1:-1]) == HALL_LEN:
            inner = ln[1:-1]
            hallway = tuple(c if c in 'ABCD' else '.' for c in inner)
            break

    # Room lines: rows containing letters/dots at columns 3,5,7,9
    room_lines: List[str] = []
    for ln in lines:
        cols = []
        for idx in (3, 5, 7, 9):
            if 0 <= idx < len(ln):
                cols.append(ln[idx])
        if len(cols) == 4 and any(c in 'ABCD' for c in cols):
            room_lines.append(ln)

    depth = len(room_lines)
    if depth not in (2, 4):
        # Fallback to the common cases; input should be 2 or 4
        depth = 2 if depth <= 2 else 4
        room_lines = room_lines[:depth]

    # Build rooms as stacks (bottom -> top), variable length (only occupants, no '.')
    rooms_stack: List[List[str]] = [[], [], [], []]
    # room_lines currently top..bottom; iterate from bottom up so we push bottom first
    for row in reversed(room_lines):
        for ridx, cidx in enumerate((3, 5, 7, 9)):
            if cidx < len(row) and row[cidx] in 'ABCD':
                rooms_stack[ridx].append(row[cidx])

    rooms = tuple(tuple(st) for st in rooms_stack)
    return hallway, rooms, depth

# ---------------- Helpers ----------------
def is_room_complete(room: Tuple[str, ...], ridx: int, depth: int) -> bool:
    """Check if room is completely filled with correct type."""
    t = TARGET[ridx]
    return len(room) == depth and all(a == t for a in room)

def is_room_ready(room: Tuple[str, ...], ridx: int) -> bool:
    """Check if room can accept its type (contains only its type or is empty)."""
    t = TARGET[ridx]
    return all(a == t for a in room)

def hallway_path_clear(hall: Tuple[str, ...], i: int, j: int) -> bool:
    """
    Check if path from i to j is clear.
    Path is inclusive of j, exclusive of i.
    """
    if i == j: 
        return True
    step = 1 if j > i else -1
    k = i + step
    while True:
        if hall[k] != '.':
            return False
        if k == j:
            return True
        k += step

def heuristic(state: State, depth: int) -> int:
    """
    Admissible heuristic: lower bound on cost to reach goal.
    Ignores all blocking and assumes direct paths.
    """
    hall, rooms = state
    h = 0
    
    # Hallway amphipods: minimal |dx| + 1 to step into target room
    for pos, a in enumerate(hall):
        if a == '.':
            continue
        ridx = TYPE_TO_ROOM[a]
        tgt = ROOM_POS[ridx]
        h += (abs(pos - tgt) + 1) * COSTS[a]

    # Amphipods in rooms: minimal (exit) + (hallway) + (enter)
    for ridx, room in enumerate(rooms):
        t = TARGET[ridx]
        for idx_from_bottom, a in enumerate(room):
            # Skip amphipods already in correct position with correct types below
            if a == t and all(x == t for x in room[:idx_from_bottom]):
                continue
            
            # Calculate minimum steps to target room
            # Steps to exit: from position idx_from_bottom to hallway
            exit_steps = depth - idx_from_bottom
            hall_dist = abs(ROOM_POS[ridx] - ROOM_POS[TYPE_TO_ROOM[a]])
            enter_steps = 1  # Minimum one step into target room
            h += (exit_steps + hall_dist + enter_steps) * COSTS[a]
    
    return h

# ---------------- Move generation ----------------
def neighbors(state: State, depth: int) -> Iterable[Tuple[int, State]]:
    """
    Generate all valid next states from current state.
    
    Yields:
        (cost, next_state) tuples
    """
    hall, rooms = state
    next_states: List[Tuple[int, State]] = []
    made_h2r = False  # Track if any hallway->room move exists

    # Priority 1: Hallway -> target room (only allowed move for hallway occupants)
    for hi, a in enumerate(hall):
        if a == '.':
            continue
        ridx = TYPE_TO_ROOM[a]
        room = rooms[ridx]
        
        # Room must have space and contain only correct type
        if len(room) < depth and is_room_ready(room, ridx):
            door = ROOM_POS[ridx]
            if hallway_path_clear(hall, hi, door):
                # Calculate cost: hallway distance + room depth
                room_distance = depth - len(room)
                steps = abs(hi - door) + room_distance
                cost = steps * COSTS[a]

                # Create new state
                new_hall = list(hall)
                new_hall[hi] = '.'
                new_rooms = [list(r) for r in rooms]
                new_rooms[ridx].append(a)
                next_states.append((cost, (tuple(new_hall), tuple(tuple(x) for x in new_rooms))))
                made_h2r = True

    # Optimization: if hallway->room move exists, skip room->hallway moves
    # This reduces branching factor significantly
    if made_h2r:
        return next_states

    # Priority 2: Room -> hallway (only if room needs to be cleared)
    for ridx in range(4):
        room = rooms[ridx]
        if not room:
            continue
        
        # Skip if room is complete or ready (no need to move out)
        if is_room_complete(room, ridx, depth) or is_room_ready(room, ridx):
            continue
        
        a = room[-1]  # Top occupant
        door = ROOM_POS[ridx]
        exit_steps = depth - len(room) + 1  # Steps from top to hallway
        
        # Scan left from door
        k = door - 1
        while k >= 0 and hall[k] == '.':
            if k not in FORBIDDEN_STOPS:
                steps = exit_steps + (door - k)
                cost = steps * COSTS[a]
                new_hall = list(hall)
                new_hall[k] = a
                new_rooms = [list(r) for r in rooms]
                new_rooms[ridx] = list(room[:-1])
                next_states.append((cost, (tuple(new_hall), tuple(tuple(x) for x in new_rooms))))
            k -= 1
        
        # Scan right from door
        k = door + 1
        while k < HALL_LEN and hall[k] == '.':
            if k not in FORBIDDEN_STOPS:
                steps = exit_steps + (k - door)
                cost = steps * COSTS[a]
                new_hall = list(hall)
                new_hall[k] = a
                new_rooms = [list(r) for r in rooms]
                new_rooms[ridx] = list(room[:-1])
                next_states.append((cost, (tuple(new_hall), tuple(tuple(x) for x in new_rooms))))
            k += 1

    return next_states

# ---------------- A* search ----------------
def solve(lines: List[str]) -> int:
    """
    Solve the amphipod sorting puzzle using A* search.
    
    Args:
        lines: Input lines representing the puzzle
        
    Returns:
        Minimum energy cost to reach goal state
    """
    hallway, rooms, depth = parse(lines)
    start: State = (hallway, rooms)
    goal: State = (tuple('.' for _ in range(HALL_LEN)),
                   (tuple('A' for _ in range(depth)),
                    tuple('B' for _ in range(depth)),
                    tuple('C' for _ in range(depth)),
                    tuple('D' for _ in range(depth))))

    # A* with admissible heuristic guarantees optimal solution
    gbest: Dict[State, int] = {start: 0}
    heap = []
    heappush(heap, (heuristic(start, depth) + 0, 0, start))

    while heap:
        f, g, s = heappop(heap)
        
        # Skip outdated entries
        if g != gbest.get(s, INF):
            continue
            
        # Goal reached
        if s == goal:
            return g
        
        # Explore neighbors
        for move_cost, nxt in neighbors(s, depth):
            ng = g + move_cost
            if ng < gbest.get(nxt, INF):
                gbest[nxt] = ng
                heappush(heap, (ng + heuristic(nxt, depth), ng, nxt))

    return -1  # Should not happen with valid input

# ---------------- Testing ----------------
def run_tests():
    """Run test cases to verify correctness."""
    # Test case 1: Depth 2
    test1 = [
        "#############",
        "#...........#",
        "###B#C#B#D###",
        "  #A#D#C#A#",
        "  #########"
    ]
    result1 = solve(test1)
    assert result1 == 12521, f"Test 1 failed: expected 12521, got {result1}"
    print(f"✓ Test 1 passed: {result1}")
    
    # Test case 2: Depth 4
    test2 = [
        "#############",
        "#...........#",
        "###B#C#B#D###",
        "  #D#C#B#A#",
        "  #D#B#A#C#",
        "  #A#D#C#A#",
        "  #########"
    ]
    result2 = solve(test2)
    assert result2 == 44169, f"Test 2 failed: expected 44169, got {result2}"
    print(f"✓ Test 2 passed: {result2}")
    
    print("\n🎉 All tests passed! Solution is correct.")

def main():
    """Main entry point."""
    # Check if running tests
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        run_tests()
        return
    
    # Read input and solve
    lines = [ln.rstrip('\n') for ln in sys.stdin]
    print(solve(lines))

if __name__ == "__main__":
    main()# submit
# nudge
