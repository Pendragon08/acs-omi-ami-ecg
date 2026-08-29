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
    # Fairseq writes both prefixed epoch aggregates (train_*/valid_*) and a
    # compact train record (loss/acs_loss/omi_loss).  Accept both forms.
    $isTrain = (($names -match '^train_') -or ($names -contains 'loss'))
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
        total_loss = if ($phase -eq 'train') { if ($null -ne $record.train_loss) { $record.train_loss } else { $record.loss } } else { $record.valid_loss }
        acs_loss = if ($phase -eq 'train') { if ($null -ne $record.train_acs_loss) { $record.train_acs_loss } else { $record.acs_loss } } else { $record.valid_acs_loss }
        omi_loss = if ($phase -eq 'train') { if ($null -ne $record.train_omi_loss) { $record.train_omi_loss } else { $record.omi_loss } } else { $record.valid_omi_loss }
        acs_accuracy = if ($phase -eq 'train') { if ($null -ne $record.train_acs_accuracy) { $record.train_acs_accuracy } else { $record.acs_accuracy } } else { $record.valid_acs_accuracy }
        omi_accuracy = if ($phase -eq 'train') { if ($null -ne $record.train_omi_accuracy) { $record.train_omi_accuracy } else { $record.omi_accuracy } } else { $record.valid_omi_accuracy }
        learning_rate = if ($null -ne $record.train_lr) { $record.train_lr } else { $record.lr }
        best_loss = $record.valid_best_loss
    }
}

$parent = Split-Path -Parent $OutputCsv
New-Item -ItemType Directory -Force -Path $parent | Out-Null
$rows | Export-Csv -LiteralPath $OutputCsv -NoTypeInformation -Encoding UTF8
Write-Output "wrote $($rows.Count) deduplicated rows to $OutputCsv"
