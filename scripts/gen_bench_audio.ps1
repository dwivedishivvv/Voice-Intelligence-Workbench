Add-Type -AssemblyName System.Speech

$lines = @(
  @{v='Microsoft David Desktop'; t='Good morning everyone, thanks for joining the call today. I want to start by going over the quarterly numbers before we get into the roadmap discussion.'},
  @{v='Microsoft Zira Desktop';  t='Sounds good. Before that, can I just flag that the dashboard outage from Tuesday is still being investigated by the infrastructure team.'},
  @{v='Microsoft David Desktop'; t='Yes, I saw that thread. Let us make sure we circle back to it at the end. For now, revenue was up twelve percent compared to last quarter, driven mostly by the enterprise segment.'},
  @{v='Microsoft Zira Desktop';  t='That is great news. Did the new pricing tier contribute much to that, or was it mostly existing customers expanding their usage.'},
  @{v='Microsoft David Desktop'; t='A bit of both actually. About forty percent came from upsells on the existing accounts, and the rest came from new logos signed in the last six weeks.'},
  @{v='Microsoft Zira Desktop';  t='Nice. I think the sales team deserves a shout out for that. Should we talk about churn as well, since that has been a concern.'},
  @{v='Microsoft David Desktop'; t='Right, churn dropped slightly this quarter, down to about four percent, which is the lowest it has been all year. The support team improvements are clearly helping.'},
  @{v='Microsoft Zira Desktop';  t='Good to hear. Switching gears, what is the latest on the roadmap for next quarter. I know engineering has been heads down on the new transcription pipeline.'},
  @{v='Microsoft David Desktop'; t='Yes, the pipeline rewrite is close to done. We are targeting a beta release in three weeks, and the diarization accuracy has already improved noticeably in our internal tests.'},
  @{v='Microsoft Zira Desktop';  t='That is exciting. Will that beta be available to all customers or just a limited rollout to start.'},
  @{v='Microsoft David Desktop'; t='Just a limited rollout initially, probably to about ten design partners, so we can gather feedback before the wider release.'},
  @{v='Microsoft Zira Desktop';  t='Makes sense. Okay, let us circle back to that dashboard outage now since we have a few minutes left.'},
  @{v='Microsoft David Desktop'; t='Sure. The root cause looks like a database connection pool exhaustion issue under heavy load. A fix is already being tested in staging.'},
  @{v='Microsoft Zira Desktop';  t='Good, I will let the customer success team know so they can update anyone who filed a ticket about it. Thanks everyone, that is all from me.'},
  @{v='Microsoft David Desktop'; t='Thanks all, talk soon.'}
)

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$outDir = "C:/data/bench"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$parts = @()
for ($i = 0; $i -lt $lines.Count; $i++) {
  $p = Join-Path $outDir ("part_{0:D2}.wav" -f $i)
  $synth.SelectVoice($lines[$i].v)
  $synth.SetOutputToWaveFile($p, (New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)))
  $synth.Speak($lines[$i].t)
  $synth.SetOutputToNull()
  $parts += $p
}
Write-Host ("generated {0} parts" -f $parts.Count)
$parts | Out-File -Encoding ascii -FilePath (Join-Path $outDir "parts.txt")
