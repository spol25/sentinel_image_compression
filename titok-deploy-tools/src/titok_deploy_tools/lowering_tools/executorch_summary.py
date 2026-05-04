from collections import Counter
from typing import Any

from executorch.exir.schema import DelegateCall, KernelCall


def _normalize_operator_name(name: str, overload: str) -> str:
    return f"{name}.{overload}" if overload else name


def _top_counts(counter: Counter[str]) -> list[dict[str, Any]]:
    return [{"name": name, "count": count} for name, count in counter.most_common()]


def summarize_executorch_program(program_manager_or_program: Any) -> dict[str, Any]:
    """Return a compact runtime-facing summary of an ExecuTorch program.

    The returned structure is intentionally JSON-friendly so lowering scripts
    can persist it directly inside their summary artifact.
    """

    if hasattr(program_manager_or_program, "executorch_program"):
        program = program_manager_or_program.executorch_program
    else:
        program = program_manager_or_program

    total_kernel_ops: Counter[str] = Counter()
    total_delegate_calls: Counter[str] = Counter()
    total_instruction_kinds: Counter[str] = Counter()
    plan_summaries: list[dict[str, Any]] = []

    for plan_index, plan in enumerate(program.execution_plan):
        plan_kernel_ops: Counter[str] = Counter()
        plan_delegate_calls: Counter[str] = Counter()
        plan_instruction_kinds: Counter[str] = Counter()
        chain_summaries: list[dict[str, Any]] = []

        for chain_index, chain in enumerate(plan.chains):
            chain_kernel_ops: Counter[str] = Counter()
            chain_delegate_calls: Counter[str] = Counter()
            chain_instruction_kinds: Counter[str] = Counter()

            for instruction in chain.instructions:
                instr_args = instruction.instr_args
                if isinstance(instr_args, KernelCall):
                    op = plan.operators[instr_args.op_index]
                    op_name = _normalize_operator_name(op.name, op.overload)
                    chain_kernel_ops[op_name] += 1
                    chain_instruction_kinds["kernel_call"] += 1
                elif isinstance(instr_args, DelegateCall):
                    delegate = plan.delegates[instr_args.delegate_index]
                    chain_delegate_calls[delegate.id] += 1
                    chain_instruction_kinds["delegate_call"] += 1
                else:
                    chain_instruction_kinds[type(instr_args).__name__] += 1

            plan_kernel_ops.update(chain_kernel_ops)
            plan_delegate_calls.update(chain_delegate_calls)
            plan_instruction_kinds.update(chain_instruction_kinds)
            chain_summaries.append(
                {
                    "chain_index": chain_index,
                    "instruction_counts": dict(chain_instruction_kinds),
                    "kernel_op_counts": _top_counts(chain_kernel_ops),
                    "delegate_call_counts": _top_counts(chain_delegate_calls),
                }
            )

        total_kernel_ops.update(plan_kernel_ops)
        total_delegate_calls.update(plan_delegate_calls)
        total_instruction_kinds.update(plan_instruction_kinds)
        plan_summaries.append(
            {
                "plan_index": plan_index,
                "plan_name": plan.name,
                "instruction_counts": dict(plan_instruction_kinds),
                "kernel_op_counts": _top_counts(plan_kernel_ops),
                "delegate_call_counts": _top_counts(plan_delegate_calls),
                "chains": chain_summaries,
            }
        )

    return {
        "instruction_counts": dict(total_instruction_kinds),
        "kernel_op_counts": _top_counts(total_kernel_ops),
        "delegate_call_counts": _top_counts(total_delegate_calls),
        "plans": plan_summaries,
    }
