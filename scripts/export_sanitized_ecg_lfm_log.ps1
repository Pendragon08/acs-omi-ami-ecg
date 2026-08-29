param(
    [Parameter(Mandatory = $true)][string]$InputJsonl,
    [Parameter(Mandatory = $true)][string]$OutputCsv,
    [string]$RunName = 'ecg_lfm'
)

$seen = @{}
$rows = @()
foreach ($line in Get-Content -LiteralPath $InputJsonl -Encoding UTF8) {
    try { $record = $line | ConvertFrom-Json } catch { continue }
    $names = @($record.PSObject.Properties.Name)
    $isTrain = $names -match '^train_'
    $isValid = $names -match '^valid_'
    if (-not ($isTrain -or $isValid)) { continue }
    $phase = if ($isValid) { 'valid' } else { 'train' }
    $key = "$phase-$($record.epoch)"
    if ($seen.ContainsKey($key)) { continue }
    $seen[$key] = $true
    $rows += [pscustomobject][ordered]@{
        run = $RunName
        phase = $phase
        epoch = $record.epoch
        total_loss = if ($phase -eq 'train') { $record.train_loss } else { $record.valid_loss }
        acs_loss = if ($phase -eq 'train') { $record.train_acs_loss } else { $record.valid_acs_loss }
        omi_loss = if ($phase -eq 'train') { $record.train_omi_loss } else { $record.valid_omi_loss }
        acs_accuracy = if ($phase -eq 'train') { $record.train_acs_accuracy } else { $record.valid_acs_accuracy }
        omi_accuracy = if ($phase -eq 'train') { $record.train_omi_accuracy } else { $record.valid_omi_accuracy }
        learning_rate = $record.train_lr
        best_loss = $record.valid_best_loss
    }
}

$parent = Split-Path -Parent $OutputCsv
New-Item -ItemType Directory -Force -Path $parent | Out-Null
$rows | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation -Encoding UTF8
Write-Output "wrote $($rows.Count) deduplicated rows to $OutputCsv"
