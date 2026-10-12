<#
V1.2 六阶段统一工程验收入口；由 Python 执行逐项检查并保存 JSON/Markdown 和完整日志。
使用已有 langgraph 环境，不安装依赖、不接入实际业务库；任一未通过时退出非零。
#>
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repoRoot
try {
    & conda run --no-capture-output -n langgraph python scripts/engineering_acceptance.py @args
    $checkExitCode = $LASTEXITCODE
}
finally { Pop-Location }
exit $checkExitCode
