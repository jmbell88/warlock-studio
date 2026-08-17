Set-Location D:\Projects\Warlock
uv run python -m warlock.bench run --recipe sdxl-cfg-raw --seeds 42,1337
uv run python -m warlock.bench run --recipe sdxl-cfg-pag-raw --seeds 42,1337
uv run python -m warlock.bench run --recipe klein-distilled-raw --seeds 42,1337
"BENCH-ALL-DONE"
