from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.tosa import TosaSpecification


class EthosUCompatCompileSpec(EthosUCompileSpec):
    """Compatibility wrapper adding U65 target handling for older ExecuTorch builds."""

    def __init__(
        self,
        target: str,
        system_config: str | None = None,
        memory_mode: str | None = None,
        extra_flags: list[str] | None = None,
        config_ini: str | None = "Arm/vela.ini",
    ):
        target_lower = target.lower()
        if "ethos-u65" not in target_lower:
            super().__init__(
                target=target,
                system_config=system_config,
                memory_mode=memory_mode,
                extra_flags=extra_flags,
                config_ini=config_ini,
            )
            return

        self.target = target
        if config_ini is None:
            config_ini = "Arm/vela.ini"
        compiler_flags = [] if extra_flags is None else list(extra_flags)
        compiler_flags.extend(
            [
                f"--accelerator-config={target}",
                f"--config={config_ini}",
                "--output-format=raw",
                "--debug-force-regor",
            ]
        )

        # NXP i.MX93 U65 guidance uses Ethos_U65_High_End with SRAM-only mode.
        if system_config is None:
            system_config = "Ethos_U65_High_End"
        if memory_mode is None:
            memory_mode = "Sram_Only"

        compiler_flags.append(f"--system-config={system_config}")
        compiler_flags.append(f"--memory-mode={memory_mode}")

        tosa_spec = TosaSpecification.create_from_string("TOSA-1.0+INT+int16+int4")
        self._set_compile_specs(tosa_spec, compiler_flags)
        self.validate()
