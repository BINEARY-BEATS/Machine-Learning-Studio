"""Command Line Interface for ML Studio."""

import argparse
import json
import sys
from pathlib import Path
from ml_studio.api import Project

def main():
    parser = argparse.ArgumentParser(prog="mls", description="Machine Learning Studio CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Project commands
    project_parser = subparsers.add_parser("project", help="Manage projects")
    project_sub = project_parser.add_subparsers(dest="subcommand", required=True)
    
    create_parser = project_sub.add_parser("create", help="Create a new project")
    create_parser.add_argument("name", help="Project name")
    
    open_parser = project_sub.add_parser("open", help="Open a project")
    open_parser.add_argument("name", help="Project name")
    
    target_parser = project_sub.add_parser("set-target", help="Set target column")
    target_parser.add_argument("name", help="Project name")
    target_parser.add_argument("target", help="Target column name")
    
    info_parser = project_sub.add_parser("info", help="Show project info")
    info_parser.add_argument("name", help="Project name")
    
    list_parser = project_sub.add_parser("list", help="List all projects")

    # Data commands
    data_parser = subparsers.add_parser("data", help="Manage data")
    data_sub = data_parser.add_subparsers(dest="subcommand", required=True)
    
    load_parser = data_sub.add_parser("load", help="Load data into a project")
    load_parser.add_argument("source", help="File path or URI")
    load_parser.add_argument("--project", required=True, help="Project name")
    
    head_parser = data_sub.add_parser("head", help="Show first N rows of data")
    head_parser.add_argument("project", help="Project name")
    head_parser.add_argument("-n", type=int, default=10, help="Number of rows")
    
    data_info_parser = data_sub.add_parser("info", help="Show dataset info")
    data_info_parser.add_argument("project", help="Project name")

    # Profile command
    prof_parser = subparsers.add_parser("profile", help="Profile project dataset")
    prof_parser.add_argument("project", help="Project name")
    prof_parser.add_argument("--out", help="Output file path for JSON profile")
    prof_parser.add_argument("--json", action="store_true", help="Print raw JSON to stdout")

    args = parser.parse_args()

    if args.command == "project":
        if args.subcommand == "create":
            p = Project.create(args.name)
            p.save()
            print(f"Project '{args.name}' created.")
        elif args.subcommand == "open":
            p = Project.open(args.name)
            print(f"Project '{args.name}' opened successfully.")
            if p.dataset:
                print(f"Dataset: {p.dataset.row_count} rows")
        elif args.subcommand == "set-target":
            p = Project.open(args.name)
            p.set_target(args.target)
            p.save()
            print(f"Target set to '{args.target}' for project '{args.name}'.")
        elif args.subcommand == "info":
            print(f"Not yet implemented (planned for Phase 2): Project info for {args.name}")
        elif args.subcommand == "list":
            for d in Path(".").iterdir():
                if d.is_dir() and (d / "project.json").exists():
                    print(f"- {d.name}")

    elif args.command == "data":
        if args.subcommand == "load":
            p = Project.open(args.project)
            p.load_data(args.source)
            print(f"Loaded {p.dataset.row_count} rows into '{args.project}'.")
        elif args.subcommand == "head":
            print(f"Not yet implemented (planned for Phase 2): Data head for {args.project} (-n {args.n})")
        elif args.subcommand == "info":
            print(f"Not yet implemented (planned for Phase 2): Data info for {args.project}")

    elif args.command == "profile":
        p = Project.open(args.project)
        prof = p.profile()
        if args.json:
            import json
            print(json.dumps(prof, indent=2))
        elif args.out:
            import json
            with open(args.out, "w") as f:
                json.dump(prof, f, indent=2)
            print(f"Profile saved to {args.out}")
        else:
            print(f"Profile complete. Quality Score: {prof.get('quality_score', 0):.2f}")

if __name__ == "__main__":
    main()
