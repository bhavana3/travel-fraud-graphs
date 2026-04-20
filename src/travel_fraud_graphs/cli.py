"""
Command-line interface for travel_fraud_graphs.

Usage examples
--------------
# Generate medium dataset, export CSV + stats report
tfg generate --scale medium --seed 42 --output ./tfg_output --format csv

# Generate small dataset and print stats only
tfg generate --scale small --seed 0 --stats-only

# Show dataset statistics for existing CSV output
tfg stats ./tfg_output
"""

import argparse
import sys
import os
import time
from pathlib import Path


def cmd_generate(args):
    from travel_fraud_graphs import generate
    from travel_fraud_graphs.stats import compute_stats, format_report, save_report

    print(f"\n[travel_fraud_graphs]  Generating '{args.scale}' dataset  (seed={args.seed}) ...")
    t0 = time.time()

    override = {}
    if args.n_users:
        override["n_users"] = args.n_users
    if args.ticketing_rings:
        override["n_ticketing_rings"] = args.ticketing_rings
    if args.ghost_hotel_rings:
        override["n_ghost_hotel_rings"] = args.ghost_hotel_rings
    if args.ato_rings:
        override["n_ato_rings"] = args.ato_rings

    data = generate(scale=args.scale, seed=args.seed, **override)
    elapsed = time.time() - t0
    print(f"  Generated in {elapsed:.1f}s")
    print(f"  Users: {data.metadata['n_users_total']:,}  "
          f"Bookings: {data.metadata['n_bookings_total']:,}  "
          f"Hotels: {data.metadata['n_hotels_total']:,}  "
          f"Flights: {data.metadata['n_flights_total']:,}")

    # Stats
    stats = compute_stats(data)
    print("\n" + format_report(stats))

    if args.stats_only:
        return

    # Export
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    fmt = args.format.lower()

    if fmt in ("csv", "all"):
        from travel_fraud_graphs.exporters import export_csv
        csv_dir = output / "csv"
        export_csv(data, csv_dir)
        print(f"\n  CSV written to: {csv_dir}")

    if fmt in ("networkx", "all"):
        try:
            from travel_fraud_graphs.exporters import export_networkx
            import pickle
            G = export_networkx(data)
            nx_path = output / "graph.networkx.pkl"
            with open(nx_path, "wb") as f:
                pickle.dump(G, f)
            print(f"  NetworkX graph written to: {nx_path}")
        except ImportError as e:
            print(f"  [skip] NetworkX export: {e}")

    if fmt in ("pyg", "all"):
        try:
            from travel_fraud_graphs.exporters import export_pyg
            import torch
            hetero = export_pyg(data)
            pyg_path = output / "graph.pt"
            torch.save(hetero, pyg_path)
            print(f"  PyG HeteroData written to: {pyg_path}")
        except ImportError as e:
            print(f"  [skip] PyG export: {e}")

    if fmt in ("dgl", "all"):
        try:
            from travel_fraud_graphs.exporters import export_dgl
            import dgl
            g = export_dgl(data)
            dgl_path = output / "graph.dgl"
            dgl.save_graphs(str(dgl_path), [g])
            print(f"  DGL graph written to: {dgl_path}")
        except ImportError as e:
            print(f"  [skip] DGL export: {e}")

    # Save stats report
    save_report(stats, str(output / "stats_report"))
    print(f"  Stats report written to: {output / 'stats_report.txt'}")
    print("\nDone.\n")


def cmd_stats(args):
    """Read an existing CSV export and print stats."""
    import json
    meta_path = Path(args.directory) / "metadata.json"
    if not meta_path.exists():
        print(f"Error: metadata.json not found in {args.directory}")
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)
    print(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(
        prog="tfg",
        description="Travel Fraud Graph Generator",
    )
    subparsers = parser.add_subparsers(dest="command")

    # generate sub-command
    gen = subparsers.add_parser("generate", help="Generate a new dataset")
    gen.add_argument("--scale",  default="medium",
                     choices=["toy", "small", "medium", "large", "xlarge"],
                     help="Dataset scale preset")
    gen.add_argument("--seed",   type=int, default=42,
                     help="Random seed (default: 42)")
    gen.add_argument("--output", default="./tfg_output",
                     help="Output directory (default: ./tfg_output)")
    gen.add_argument("--format", default="csv",
                     choices=["csv", "networkx", "pyg", "dgl", "all"],
                     help="Export format (default: csv)")
    gen.add_argument("--stats-only", action="store_true",
                     help="Print statistics without writing files")
    gen.add_argument("--n-users",            type=int, default=None)
    gen.add_argument("--ticketing-rings",    type=int, default=None)
    gen.add_argument("--ghost-hotel-rings",  type=int, default=None)
    gen.add_argument("--ato-rings",          type=int, default=None)

    # stats sub-command
    st = subparsers.add_parser("stats", help="Print metadata from an existing output dir")
    st.add_argument("directory", help="Path to tfg output directory")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
