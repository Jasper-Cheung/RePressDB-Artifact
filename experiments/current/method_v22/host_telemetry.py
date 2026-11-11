"""Passive host observations, started partway through the policy campaign."""
from pathlib import Path
import datetime,json,time
import psutil

here=Path(__file__).resolve().parent
with (here/'host_telemetry.jsonl').open('a',encoding='utf-8') as out:
    while True:
        completed=len(list((here/'results/cells').glob('*/summary.json')))
        vm=psutil.virtual_memory();freq=psutil.cpu_freq()
        row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_cells=completed,
                 memory_available=vm.available,memory_percent=vm.percent,
                 cpu_percent=psutil.cpu_percent(interval=None,percpu=True),reported_frequency_mhz=freq.current if freq else None,
                 note='passive observations begin during the campaign; frequency is the OS-reported value')
        out.write(json.dumps(row)+'\n');out.flush()
        if completed>=144:break
        time.sleep(10)
