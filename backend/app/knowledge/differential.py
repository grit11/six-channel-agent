"""鉴别来源的候选名称归一与显式归属拆分。

职责：保存同一名称的常用缩写，按来源候选组约束前者/后者及维度标题归属。
边界：不建立方证与病证的医学关联；名称歧义不选择候选，不读取文件或检索服务。
"""

import re

_ALIASES: dict[str, tuple[str, ...]] = {
    "葛根黄芩黄连汤": ("葛根芩连汤",),
    "麻黄杏仁甘草石膏汤": ("麻杏石甘汤", "麻杏甘石汤"),
    "茯苓桂枝白术甘草汤": ("苓桂术甘汤",),
    "茯苓桂枝甘草大枣汤": ("苓桂枣甘汤",),
    "桂枝去桂加茯苓白术汤": ("桂枝去桂加苓术汤",),
    "桂枝去芍药加蜀漆牡蛎龙骨救逆汤": ("救逆汤",),
}


def candidate_aliases(candidate: str) -> set[str]:
    """归一完整名称、证后缀及显式缩写；含病机括注时只增加括注前名称。"""
    base = re.split(r"[（(]", candidate, maxsplit=1)[0].removesuffix("证")
    bases = {base}
    # 来源显式括注中的方名可作为同名入口；病机括注不成为新候选名称。
    for bracket in re.findall(r"[（(]([^（）()]+)[）)]", candidate):
        if bracket.endswith(("汤证", "丸证", "散证")):
            bases.add(bracket.removesuffix("证"))
    for canonical, aliases in _ALIASES.items():
        if base in {canonical, *aliases}:
            bases.update((canonical, *aliases))
    return {candidate, *(name for root in bases for name in (root, root + "证"))}


def candidate_labels_match(first: str, second: str) -> bool:
    """仅比较同名/同义缩写，不把相关方证当成相同候选。"""
    return bool(candidate_aliases(first).intersection(candidate_aliases(second)))


def source_candidate_label(candidate: str, candidates: list[str]) -> str | None:
    """在当前规则组中解析唯一来源标签，精确同名优先，歧义保持未绑定。"""
    if candidate in candidates:
        return candidate
    matches = [label for label in candidates if candidate_labels_match(candidate, label)]
    return matches[0] if len(matches) == 1 else None


def _context_aliases(candidate: str) -> set[str]:
    """只在来源组内允许省略六经前缀或同组比较的腹满/下利等主题后缀。"""
    aliases = candidate_aliases(candidate)
    for name in tuple(aliases):
        name = name.removesuffix("证")
        for channel in ("太阳", "阳明", "少阳", "太阴", "少阴", "厥阴", "里"):
            if name.startswith(channel):
                short = name[len(channel) :]
                aliases.add(short)
                for suffix in ("腹满", "下利", "腹痛", "消渴", "咽痛"):
                    if short.endswith(suffix) and len(short) > len(suffix):
                        aliases.add(short.removesuffix(suffix))
    return aliases


def _owners(prefix: str, candidates: list[str]) -> list[str]:
    """只接受唯一命名或来源明确的顺序指代，避免相似方名前缀碰撞。"""
    if prefix == "前者" and len(candidates) == 2:
        return candidates[:1]
    if prefix == "后者" and len(candidates) == 2:
        return candidates[1:]
    if prefix == "后两者" and len(candidates) == 3:
        return candidates[1:]
    matches = [item for item in candidates if prefix in _context_aliases(item)]
    if len(matches) == 1:
        return matches
    return []


def candidate_source_values(
    description: str, candidates: list[str], *, dimension: str = ""
) -> dict[str, str]:
    """从显式候选前缀或维度标题拆分原文，不把单方说明复制给双方。"""
    values: dict[str, list[str]] = {}
    prefixes = sorted(
        {alias for item in candidates for alias in _context_aliases(item)}
        | {"前者", "后者", "后两者"},
        key=len,
        reverse=True,
    )
    for segment in re.split(r"[；;]", description):
        segment = segment.strip(" 。：:")
        for prefix in prefixes:
            if not prefix or not segment.startswith(prefix):
                continue
            owners = _owners(prefix, candidates)
            value = segment[len(prefix) :].strip(" ：:")
            if owners and value:
                for owner in owners:
                    values.setdefault(owner, []).append(value)
                break
    if not values and dimension:
        owners = _owners(dimension, candidates)
        if owners:
            values[owners[0]] = [description]
    return {owner: "；".join(parts) for owner, parts in values.items()}
