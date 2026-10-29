# Downloads the Stanford 3D Scanning Repository models into data\.
# Tries graphics.stanford.edu first and falls back to a mirror of the original PLY files
# (Harvey Mudd CS155 course archive) when Stanford is unreachable.
#   powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1          # bunny, dragon, buddha
#   powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1 -Lucy    # also Lucy (~28M tris; Stanford only)
param([switch]$Lucy)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
$data = Join-Path $root "data"
$tmp = Join-Path $data "_download"
New-Item -ItemType Directory -Force $data, $tmp | Out-Null

$stanford = "http://graphics.stanford.edu/pub/3Dscanrep"
$mirror = "https://www.cs.hmc.edu/courses/2009/fall/cs155/archive/models"
$models = @(
    @{ url = "$stanford/bunny.tar.gz"; member = "bunny/reconstruction/bun_zipper.ply"; out = "bun_zipper.ply"
       mirror = "$mirror/bunny/reconstruction/bun_zipper.ply" },
    @{ url = "$stanford/dragon/dragon_recon.tar.gz"; member = "dragon_recon/dragon_vrip.ply"; out = "dragon_vrip.ply"
       mirror = "$mirror/dragon/dragon_vrip.ply" },
    @{ url = "$stanford/happy/happy_recon.tar.gz"; member = "happy_recon/happy_vrip.ply"; out = "happy_vrip.ply"
       mirror = "$mirror/happy_buddah/happy_vrip.ply" }
)
if ($Lucy) {
    $models += @{ url = "http://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz"; member = "lucy.ply"; out = "lucy.ply"; mirror = $null }
}

foreach ($m in $models) {
    $dest = Join-Path $data $m.out
    if (Test-Path $dest) { Write-Host "have $($m.out)"; continue }
    $archive = Join-Path $tmp (Split-Path $m.url -Leaf)
    try {
        Write-Host "downloading $($m.url)"
        Invoke-WebRequest -Uri $m.url -OutFile $archive -TimeoutSec 30
        tar -xzf $archive -C $tmp $m.member
        if ($LASTEXITCODE -ne 0) { throw "could not extract $($m.member)" }
        Move-Item (Join-Path $tmp $m.member) $dest
    } catch {
        if (-not $m.mirror) { Write-Warning "failed: $($_.Exception.Message) (no mirror for $($m.out))"; continue }
        Write-Host "  Stanford unavailable ($($_.Exception.Message)); trying mirror $($m.mirror)"
        Invoke-WebRequest -Uri $m.mirror -OutFile $dest
    }
    Write-Host "  -> data\$($m.out)"
}
Remove-Item -Recurse -Force $tmp
Get-ChildItem $data -Filter *.ply | Format-Table Name, @{ n = "MB"; e = { [math]::Round($_.Length / 1MB, 1) } }
