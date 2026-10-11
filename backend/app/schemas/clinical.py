"""Clinical Evidence 三值契约与冲突派生。

职责：保存事实、来源和时间，重建同时间冲突并提供当前可用事实视图。
边界：不访问外部依赖、不执行六经辨证，不以历史、推测或未解决冲突支持当前候选。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Polarity = Literal["positive", "negative", "unknown"]
FindingSource = Literal["doctor_input", "record", "image", "device", "inference"]
SixChannel = Literal["太阳", "阳明", "少阳", "太阴", "少阴", "厥阴"]

_CURRENT_TIMES = frozenset({"今日", "今天", "当前", "现在", "目前"})


def fact_time_group(time_scope: str | None) -> str:
    """归一明确的当前时间别名，其他时间原样保留，不猜测先后。"""
    return "current" if time_scope is None or time_scope in _CURRENT_TIMES else time_scope


class Finding(BaseModel):
    """单个临床事实，unknown 不得被解释为 negative。"""

    model_config = ConfigDict(extra="forbid")
    key: str = Field(min_length=1, description="标准化事实键")
    value: str | bool | float | None = Field(default=None, description="事实值")
    polarity: Polarity = Field(description="阳性、阴性或未知")
    certainty: float = Field(default=1.0, ge=0, le=1, description="抽取确定性")
    source: FindingSource = Field(description="事实来源")
    source_text: str | None = Field(default=None, description="原文证据片段")
    time_scope: str | None = Field(default=None, description="时间范围")


class Contradiction(BaseModel):
    """同一事实被明确记录为冲突时保留两侧，而不是覆盖。"""

    key: str = Field(min_length=1, description="冲突事实键")
    findings: list[Finding] = Field(min_length=2, description="互相冲突的事实")
    reason: str = Field(min_length=1, description="冲突说明")


class TimelineEvent(BaseModel):
    """医生提供的症状组起病次序；六经关联须由医生明确确认，不能按病程天数推断。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    six_channel: Literal["太阳", "阳明", "少阳", "太阴", "少阴", "厥阴"]
    finding_keys: list[str] = Field(min_length=1, description="关联的临床事实键")
    onset_order: int | None = Field(default=None, ge=0, description="相同次序表示明确同时起病")
    active_now: bool | None = Field(default=None, description="当前证候是否仍存在；缺失保持未知")
    source: FindingSource = Field(default="doctor_input")
    source_text: str = Field(min_length=1, description="医生确认的时间线原文")
    time_scope: str | None = Field(default=None, description="该组事实所属时间；历史必须明确填写")


class TreatmentEvent(BaseModel):
    """医生确认的治疗事件及前后事实引用；时间相邻不自动证明治疗因果。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    event_id: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=2000, description="已用治疗记录，非系统建议")
    event_order: int | None = Field(default=None, ge=0)
    before_finding_keys: list[str] = Field(default_factory=list)
    before_time_scope: str | None = None
    after_finding_keys: list[str] = Field(default_factory=list)
    after_time_scope: str | None = None
    source: FindingSource = "doctor_input"
    source_text: str = Field(
        min_length=1, max_length=4000, description="治疗与前后变化的医生核对依据"
    )


class RelationshipHistory(BaseModel):
    """医生核对的关系病史；空值保持未知，因果确认不能由模型代填。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    history_complete: bool | None = Field(default=None, description="发病至今病史已完整核对")
    prior_yang_phase: bool | None = Field(default=None, description="本次发病是否先经过三阳阶段")
    simultaneous_exposure: bool | None = Field(default=None, description="表里两经是否同时感邪")
    exterior_interior_pair: bool | None = Field(
        default=None, description="医生确认这两个症状组为表里两经"
    )
    no_other_current_syndrome: bool | None = Field(
        default=None, description="医生已排查其他当前证候组"
    )
    primary_six_channel: SixChannel | None = Field(default=None, description="医生核对的本证主经")
    syndrome_changed: bool | None = Field(default=None, description="医生确认脉证发生变化")
    original_pattern_applicable: bool | None = Field(default=None, description="原证型是否仍适用")
    treatment_error_confirmed: bool | None = Field(
        default=None, description="医生确认失治或误治相关，非系统因果推断"
    )
    treatment_event_ids: list[str] = Field(
        default_factory=list, description="因果核对对应的已用治疗事件 ID"
    )
    source_text: str = Field(
        default="", max_length=4000, description="医生病史核对原文，未填写不能确认关系"
    )
    source: FindingSource = Field(default="doctor_input", description="核对来源")


class ClinicalEvidence(BaseModel):
    """临床输入的结构化快照，字段缺失保持 unknown 事实而非默认阴性。"""

    model_config = ConfigDict(extra="forbid")
    chief_complaint: str | None = Field(default=None, description="主诉")
    onset: str | None = Field(default=None, description="起病")
    course: str | None = Field(default=None, description="病程")
    triggers: list[str] = Field(default_factory=list, description="诱因")
    fever_chills: dict[str, object] = Field(default_factory=dict, description="发热恶寒结构化区")
    sweating: dict[str, object] = Field(default_factory=dict, description="汗出结构化区")
    thirst_mouth: dict[str, object] = Field(default_factory=dict, description="口渴口味结构化区")
    appetite_digestive: dict[str, object] = Field(
        default_factory=dict, description="纳食消化结构化区"
    )
    stool: dict[str, object] = Field(default_factory=dict, description="大便结构化区")
    urination: dict[str, object] = Field(default_factory=dict, description="小便结构化区")
    pain: list[dict[str, object]] = Field(default_factory=list, description="疼痛结构化区")
    chest_hypochondrium: dict[str, object] = Field(default_factory=dict, description="胸胁结构化区")
    cough_dyspnea_phlegm: dict[str, object] = Field(
        default_factory=dict, description="咳喘痰结构化区"
    )
    spirit_sleep: dict[str, object] = Field(default_factory=dict, description="神志睡眠结构化区")
    limbs_body: dict[str, object] = Field(default_factory=dict, description="四肢身形结构化区")
    tongue: dict[str, object] = Field(default_factory=dict, description="舌象结构化区")
    pulse: dict[str, object] = Field(default_factory=dict, description="脉象结构化区")
    prior_treatment: list[dict[str, object]] = Field(
        default_factory=list, description="既往治疗变化"
    )
    findings: list[Finding] = Field(default_factory=list, description="结构化事实")
    contradictions: list[Contradiction] = Field(default_factory=list, description="冲突事实")
    timeline: list[TimelineEvent] = Field(default_factory=list, description="医生确认的病证时间线")
    treatment_events: list[TreatmentEvent] = Field(
        default_factory=list, description="结构化治疗前后记录；旧 prior_treatment 仅留存"
    )
    relationship_history: RelationshipHistory = Field(
        default_factory=RelationshipHistory, description="医生核对关系病史"
    )

    @model_validator(mode="after")
    def refresh_contradictions(self) -> "ClinicalEvidence":
        """按同键同时间的当前事实重建冲突，客户端列表不能覆盖真实 findings。"""
        self._rebuild_contradictions()
        return self

    def _rebuild_contradictions(self) -> None:
        """模型初始化与原地编辑均复用同一冲突推导逻辑。"""
        groups: dict[tuple[str, str], list[Finding]] = {}
        for finding in self.findings:
            if finding.polarity != "unknown" and finding.source != "inference":
                groups.setdefault((finding.key, fact_time_group(finding.time_scope)), []).append(
                    finding
                )
        self.contradictions = [
            Contradiction(
                key=key,
                findings=related,
                reason=f"同一时间范围（{time}）同时记录存在和否认，需医生核对",
            )
            for (key, time), related in groups.items()
            if {item.polarity for item in related} == {"positive", "negative"}
        ]

    def current_findings(self) -> list[Finding]:
        """返回当前可用事实；历史、推测、零确定性和冲突键不提供单向支持。"""
        self._rebuild_contradictions()
        conflicted = {
            item.key
            for item in self.contradictions
            if fact_time_group(item.findings[0].time_scope) == "current"
        }
        return [
            finding
            for finding in self.findings
            if fact_time_group(finding.time_scope) == "current"
            and finding.source != "inference"
            and finding.key not in conflicted
            and (finding.certainty > 0 or finding.polarity == "unknown")
        ]

    def by_polarity(self, polarity: Polarity) -> list[Finding]:
        """按三值极性返回独立列表，保持 unknown 与 negative 分离。"""
        return [finding for finding in self.findings if finding.polarity == polarity]

    @property
    def positive_findings(self) -> list[Finding]:
        """返回明确存在的事实。"""
        return self.by_polarity("positive")

    @property
    def negative_findings(self) -> list[Finding]:
        """返回明确否认的事实。"""
        return self.by_polarity("negative")

    @property
    def uncertain_findings(self) -> list[Finding]:
        """返回未确定事实，不能作为反证。"""
        return self.by_polarity("unknown")
