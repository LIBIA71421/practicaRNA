$ErrorActionPreference = 'Stop'

function Test-PortAvailable {
    param(
        [int]$Port
    )

    try {
        $listener = [System.Net.Sockets.TcpListener]::new(
            [System.Net.IPAddress]::Loopback,
            $Port
        )
        $listener.Start()
        $listener.Stop()
        return $true
    }
    catch {
        return $false
    }
}


try {
    $projectRoot = Split-Path -Parent $PSCommandPath
    $runsDir = Join-Path $projectRoot 'runs'

    if (-not (Test-Path $runsDir)) {
        throw "No existe el directorio de logs: $runsDir. Ejecuta primero main.py para generar datos de TensorBoard."
    }

    $port = 6006
    while (($port -lt 6016) -and (-not (Test-PortAvailable -Port $port))) {
        $port++
    }

    if ($port -ge 6016) {
        throw 'No se encontro un puerto libre entre 6006 y 6015 para iniciar TensorBoard.'
    }

    $launcherCommand = $null
    $launcherArguments = @()

    $tensorboardCandidates = @(
        (Join-Path $projectRoot '.venv\Scripts\tensorboard.exe'),
        (Join-Path $projectRoot 'venv\Scripts\tensorboard.exe')
    )

    foreach ($candidate in $tensorboardCandidates) {
        if (Test-Path $candidate) {
            $launcherCommand = $candidate
            break
        }
    }

    if (-not $launcherCommand) {
        $tensorboardCommand = Get-Command tensorboard -ErrorAction SilentlyContinue
        if ($tensorboardCommand) {
            $launcherCommand = 'tensorboard'
        }
    }

    if ($launcherCommand) {
        $launcherArguments = @(
            '--logdir',
            $runsDir,
            '--host',
            'localhost',
            '--port',
            "$port"
        )
    }
    else {
        $pythonCandidates = @(
            (Join-Path $projectRoot '.venv\Scripts\python.exe'),
            (Join-Path $projectRoot 'venv\Scripts\python.exe')
        )

        foreach ($candidate in $pythonCandidates) {
            if (Test-Path $candidate) {
                $launcherCommand = $candidate
                break
            }
        }

        if (-not $launcherCommand) {
            $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
            if ($pyLauncher) {
                $launcherCommand = 'py'
            }
        }

        if (-not $launcherCommand) {
            $pythonExe = Get-Command python -ErrorAction SilentlyContinue
            if ($pythonExe) {
                $launcherCommand = 'python'
            }
        }

        if (-not $launcherCommand) {
            throw 'No se encontro Python ni tensorboard.exe. Instala Python o usa la .venv del proyecto.'
        }

        & $launcherCommand -c "import tensorboard" *> $null
        if ($LASTEXITCODE -ne 0) {
            throw 'TensorBoard no esta instalado en el entorno detectado. Ejecuta: pip install -r requirements.txt'
        }

        $launcherArguments = @(
            '-m',
            'tensorboard.main',
            '--logdir',
            $runsDir,
            '--host',
            'localhost',
            '--port',
            "$port"
        )
    }

    $url = "http://localhost:$port"

    Write-Host "Proyecto: $projectRoot"
    Write-Host "Logs: $runsDir"
    Write-Host "Lanzador: $launcherCommand"
    Write-Host "Puerto: $port"
    Write-Host "Abriendo navegador en: $url"
    Write-Host 'TensorBoard quedara ejecutandose en esta ventana.'

    Start-Job -ScriptBlock {
        param(
            [string]$TensorBoardUrl
        )

        Start-Sleep -Seconds 3
        Start-Process $TensorBoardUrl | Out-Null
    } -ArgumentList $url | Out-Null

    & $launcherCommand @launcherArguments
}
catch {
    Write-Error $_
    Read-Host 'Presiona Enter para cerrar'
    exit 1
}







