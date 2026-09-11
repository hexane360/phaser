#!/bin/bash
# Template for slurm worker jobs. The markers below are substituted at submission time by
# `render_worker_script` (phaser/web/slurm.py).

@PREAMBLE@

python_exec=@PYTHON@

url=@URL@
echo "Running worker, connecting to '$url'"

while true; do
    "$python_exec" -m phaser worker "$url"
    sig=$(($? - 128))
    if [ $sig -gt 0 ] && [ "$(kill -l $sig)" == "HUP" ]; then
        echo "Restarting worker (SIGHUP)"
        continue
    fi
    break
done
