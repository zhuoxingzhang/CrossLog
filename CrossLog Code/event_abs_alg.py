import json
import random
import math
from typing import Dict, List, Set, Tuple


# =====================================================
# 基础工具函数
# =====================================================

def init_event_levels(system_events: Dict[str, List[str]]) -> Dict[str, int]:
    """
    初始化：每个事件当前抽象层级
    L1 -> index 0
    L2 -> index 1
    L3 -> index 2 (最抽象，初始状态)
    """
    return {event: 2 for event in system_events}


def get_system_circle(
        system_events: Dict[str, List[str]],
        event_levels: Dict[str, int]
) -> Set[str]:
    return {
        system_events[event][event_levels[event]]
        for event in system_events
    }


def compute_coupling(circle1: Set[str], circle2: Set[str]) -> float:
    """
    耦合度：
    (|I|/|C1| + |I|/|C2|) / 2
    """
    if not circle1 or not circle2:
        return 0.0
    intersection = circle1 & circle2
    return (len(intersection) / len(circle1) +
            len(intersection) / len(circle2)) / 2


# =====================================================
# 熵相关函数
# =====================================================

def entropy(circle: Set[str]) -> float:
    if not circle:
        return 0.0
    p = 1.0 / len(circle)
    return -len(circle) * p * math.log(p)


def simulate_entropy_drop(
        system_events: Dict[str, List[str]],
        event_levels: Dict[str, int],
        target_type: str
) -> float:
    """
    模拟将 target_type 向下降一级后的熵变化
    """
    old_circle = get_system_circle(system_events, event_levels)
    old_entropy = entropy(old_circle)

    simulated_levels = event_levels.copy()
    for event, levels in system_events.items():
        cur = simulated_levels[event]
        if levels[cur] == target_type and cur > 0:
            simulated_levels[event] -= 1

    new_circle = get_system_circle(system_events, simulated_levels)
    new_entropy = entropy(new_circle)

    return old_entropy - new_entropy


# =====================================================
# 抽象下降可行性判断（关键新增）
# =====================================================

def can_degrade(
        system_events: Dict[str, List[str]],
        event_levels: Dict[str, int],
        target_type: str
) -> bool:
    """
    判断某个抽象类型是否至少还能在一个事件上下降
    """
    for event, levels in system_events.items():
        cur = event_levels[event]
        if levels[cur] == target_type and cur > 0:
            return True
    return False


def degrade_events_by_type(
        system_events: Dict[str, List[str]],
        event_levels: Dict[str, int],
        target_type: str
):
    """
    实际执行下降
    """
    for event, levels in system_events.items():
        cur = event_levels[event]
        if levels[cur] == target_type and cur > 0:
            event_levels[event] -= 1


# =====================================================
# 熵最大下降选择, 如果没有可选择，则随机
# =====================================================

def select_type_to_degrade(
        events_a, levels_a,
        events_b, levels_b,
        intersection: List[str]
) -> str | None:
    """
    选择下降的抽象类型：
    1. 优先选择熵下降最大的 (事件抽象类型越不抽象，越具体的速度更快)
    2. 若所有熵下降为 0，但仍可下降，则随机选一个可下降的
    """

    best_type = None
    best_drop = -1.0

    degradable_types = []

    for t in intersection:

        # 至少在一个系统中还能下降
        if not (can_degrade(events_a, levels_a, t) or
                can_degrade(events_b, levels_b, t)):
            continue

        degradable_types.append(t)

        drop_a = simulate_entropy_drop(events_a, levels_a, t)
        drop_b = simulate_entropy_drop(events_b, levels_b, t)
        total_drop = drop_a + drop_b

        if total_drop > best_drop:
            best_drop = total_drop
            best_type = t

    if best_drop > 0:
        return best_type

    # 熵不变，但允许结构性下降（初始阶段必走这里）
    if degradable_types:
        return random.choice(degradable_types)

    return None


# =====================================================
# 主算法：两个系统的抽象耦合削减
# =====================================================

def decouple_two_systems(
        data: Dict[str, Dict[str, List[str]]],
        system_a: str,
        system_b: str,
        threshold: float,
        seed: int = 42
) -> Tuple[Dict, Set[str]]:
    random.seed(seed)

    events_a = data[system_a]
    events_b = data[system_b]

    levels_a = init_event_levels(events_a)
    levels_b = init_event_levels(events_b)

    iteration = 0

    while True:
        iteration += 1

        circle_a = get_system_circle(events_a, levels_a)
        circle_b = get_system_circle(events_b, levels_b)

        coupling = compute_coupling(circle_a, circle_b)
        print(f"[Iter {iteration}] Coupling = {coupling:.4f}")

        if coupling <= threshold:
            break

        intersection = list(circle_a & circle_b)
        if not intersection:
            print("No intersection left. Stop.")
            break

        target_type = select_type_to_degrade(
            events_a, levels_a,
            events_b, levels_b,
            intersection
        )

        # ⭐ 关键终止条件：无可下降类型
        if target_type is None:
            print("No degradable abstraction type left. Stop.")
            break

        print(f"Degrading abstraction type: {target_type}")

        degrade_events_by_type(events_a, levels_a, target_type)
        degrade_events_by_type(events_b, levels_b, target_type)

    final_result = {
        system_a: {
            event: events_a[event][levels_a[event]]
            for event in events_a
        },
        system_b: {
            event: events_b[event][levels_b[event]]
            for event in events_b
        }
    }

    used_types = (
            set(final_result[system_a].values()) |
            set(final_result[system_b].values())
    )

    return final_result, used_types


# =====================================================
# 主流程
# =====================================================

def main():
    SYSTEM_A = "halo"
    SYSTEM_B = "forum"
    TARGET_SYSTEM = "novel"
    COUPLING_THRESHOLD = 0.6

    root = r"C:/Users/zzha969/OneDrive - The University of Auckland/Desktop/UnifiedLogExp"
    INPUT_JSON = f"{root}/data/unified_level_event_abstraction.json"
    OUTPUT_JSON = f"{root}/towards_target_{TARGET_SYSTEM}_abstraction_mapping_c{COUPLING_THRESHOLD}.json"
    OUTPUT_TXT = f"{root}/towards_target_{TARGET_SYSTEM}_abstraction_pool_c{COUPLING_THRESHOLD}.txt"

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    final_result, used_types = decouple_two_systems(
        data=data,
        system_a=SYSTEM_A,
        system_b=SYSTEM_B,
        threshold=COUPLING_THRESHOLD,
        seed=42
    )

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        for t in sorted(used_types):
            f.write(t + "\n")

    print("\nDone.")
    print(f"Final abstraction JSON: {OUTPUT_JSON}")
    print(f"Used abstraction types: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
