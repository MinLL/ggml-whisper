<#
.SYNOPSIS
    Installs a minimal CUDA toolkit from NVIDIA's redistributable archives.

.DESCRIPTION
    Downloads only the components the ggml CUDA backend and the cudart archive need
    (nvcc, cudart, cuBLAS, cuFFT, CCCL and their CUDA 13 splits) instead of running the
    multi-GB installer. Every archive is verified against the sha256 in NVIDIA's
    redistrib_<version>.json manifest and merged into a standard toolkit layout at
    <Root>\v<major>.<minor>, whose folder name CMakeLists.txt uses for the archive names.

    In GitHub Actions it exports CUDA_PATH / CUDA_PATH_V<major>_<minor> to GITHUB_ENV and
    the bin folders to GITHUB_PATH.
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Version,

    [string]$Root = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",

    # Components missing from a release's manifest are skipped (CUDA 12 ships cuda_cccl,
    # CUDA 13 ships cccl + cuda_crt + libnvvm), so one list serves both majors.
    [string[]]$Components = @(
        "cuda_nvcc",
        "cuda_cudart",
        "cuda_crt",
        "libnvvm",
        "cuda_cccl",
        "cccl",
        "cuda_nvtx",
        "cuda_profiler_api",
        "libcublas",
        "libcufft"
    )
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$baseUrl = "https://developer.download.nvidia.com/compute/cuda/redist"
$parts = $Version.Split(".")
if ($parts.Count -lt 2) { throw "CUDA version must be <major>.<minor>[.<patch>], got '$Version'" }
$shortVersion = "$($parts[0]).$($parts[1])"
$cudaPath = Join-Path $Root "v$shortVersion"

$work = Join-Path ([System.IO.Path]::GetTempPath()) "cuda-redist-$Version"
New-Item -ItemType Directory -Force -Path $work, $cudaPath | Out-Null

Write-Host "Fetching manifest redistrib_$Version.json"
$manifest = Invoke-RestMethod -Uri "$baseUrl/redistrib_$Version.json"

$sevenZip = (Get-Command 7z -ErrorAction SilentlyContinue).Source

foreach ($name in $Components) {
    $entry = $manifest.$name
    if (-not $entry -or -not $entry.'windows-x86_64') {
        Write-Host "  - ${name}: not in CUDA $Version for windows-x86_64, skipped"
        continue
    }
    $pkg = $entry.'windows-x86_64'
    $url = "$baseUrl/$($pkg.relative_path)"
    $zip = Join-Path $work (Split-Path $pkg.relative_path -Leaf)
    Write-Host "  - $name $($entry.version) ($([math]::Round($pkg.size / 1MB)) MB)"

    $attempt = 0
    while ($true) {
        $attempt++
        try {
            Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
            $hash = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLowerInvariant()
            if ($hash -ne $pkg.sha256) { throw "sha256 mismatch for $url (got $hash, want $($pkg.sha256))" }
            break
        } catch {
            if ($attempt -ge 3) { throw }
            Write-Warning "Download attempt $attempt failed: $_. Retrying."
            Start-Sleep -Seconds (10 * $attempt)
        }
    }

    $extract = Join-Path $work $name
    if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
    if ($sevenZip) {
        & $sevenZip x -y -bso0 -bsp0 "-o$extract" $zip
        if ($LASTEXITCODE -ne 0) { throw "7z failed to extract $zip" }
    } else {
        Expand-Archive -Path $zip -DestinationPath $extract
    }
    Remove-Item -Force $zip

    # Each archive holds one top-level folder (<component>-windows-x86_64-<ver>-archive)
    # with bin/include/lib/... below it; merge that into the toolkit root.
    $top = Get-ChildItem -Path $extract -Directory | Select-Object -First 1
    & robocopy $top.FullName $cudaPath /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed merging $name (exit $LASTEXITCODE)" }
    $global:LASTEXITCODE = 0
    Remove-Item -Recurse -Force $extract
}

$nvcc = Join-Path $cudaPath "bin\nvcc.exe"
if (-not (Test-Path $nvcc)) { throw "nvcc.exe not found at $nvcc after install" }
& $nvcc --version
if ($LASTEXITCODE -ne 0) { throw "nvcc --version failed" }

$binDirs = @((Join-Path $cudaPath "bin"))
$x64Bin = Join-Path $cudaPath "bin\x64"
if (Test-Path $x64Bin) { $binDirs += $x64Bin }

if ($env:GITHUB_ENV) {
    $versionVar = "CUDA_PATH_V$($parts[0])_$($parts[1])"
    "CUDA_PATH=$cudaPath" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
    "$versionVar=$cudaPath" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
}
if ($env:GITHUB_PATH) {
    $binDirs | Out-File -FilePath $env:GITHUB_PATH -Append -Encoding utf8
}
Write-Host "CUDA $Version installed at $cudaPath"
