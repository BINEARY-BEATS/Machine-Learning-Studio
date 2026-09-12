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
    
    set_role_parser = project_sub.add_parser("set-role", help="Set role for a column")
    set_role_parser.add_argument("name", help="Project name")
    set_role_parser.add_argument("column", help="Column name")
    set_role_parser.add_argument("role", help="Role (feature, target, id, group, time_index, weight, drop)")
    
    roles_parser = project_sub.add_parser("roles", help="Show all column roles")
    roles_parser.add_argument("name", help="Project name")
    
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

    # Pipeline commands
    pipe_parser = subparsers.add_parser("pipeline", help="Manage pipelines")
    pipe_sub = pipe_parser.add_subparsers(dest="subcommand", required=True)
    
    pipe_create_parser = pipe_sub.add_parser("create", help="Create a pipeline")
    pipe_create_parser.add_argument("project", help="Project name")
    pipe_create_parser.add_argument("--recipe", help="Recipe name")
    pipe_create_parser.add_argument("--from-yaml", help="Path to yaml file")
    
    pipe_show_parser = pipe_sub.add_parser("show", help="Show pipeline summary")
    pipe_show_parser.add_argument("project", help="Project name")
    
    pipe_preview_parser = pipe_sub.add_parser("preview", help="Preview pipeline shape changes")
    pipe_preview_parser.add_argument("project", help="Project name")
    
    pipe_apply_parser = pipe_sub.add_parser("apply", help="Apply pipeline to project data")
    pipe_apply_parser.add_argument("project", help="Project name")

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
            p = Project.open(args.name)
            print(f"Project: {args.name}")
            print(f"Task: {p.task}")
            print(f"Target: {p.target}")
            if p.dataset:
                print(f"Dataset: {p.dataset.row_count} rows, {len(p.dataset.dataframe.columns)} columns")
            if p.pipeline:
                print(f"Pipeline: {len(p.pipeline.steps)} steps")
        elif args.subcommand == "list":
            for d in Path(".").iterdir():
                if d.is_dir() and (d / "project.json").exists():
                    print(f"- {d.name}")
        elif args.subcommand == "set-role":
            p = Project.open(args.name)
            p.set_role(args.column, args.role)
            print(f"Set role of '{args.column}' to '{args.role}'.")
        elif args.subcommand == "roles":
            p = Project.open(args.name)
            import json
            print(json.dumps(p.schema_columns, indent=2))

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

    elif args.command == "pipeline":
        p = Project.open(args.project)
        if args.subcommand == "create":
            if args.recipe:
                from ml_studio.core.recipes import apply_recipe
                pipeline = apply_recipe(args.recipe, p.schema_columns)
            elif args.from_yaml:
                from ml_studio.core.pipeline import Pipeline
                with open(args.from_yaml, "r") as f:
                    pipeline = Pipeline.from_yaml(f.read())
            else:
                print("Must provide either --recipe or --from-yaml")
                sys.exit(1)
            p.pipeline = pipeline
            p.save()
            print(f"Pipeline created and saved to project '{args.project}'.")
        elif args.subcommand == "show":
            if not p.pipeline:
                print("No pipeline exists for this project.")
                sys.exit(1)
            print(p.pipeline.describe())
            print(f"\nHash: {p.pipeline.hash()}")
        elif args.subcommand == "preview":
            if not p.pipeline:
                print("No pipeline exists for this project.")
                sys.exit(1)
            X, y = p.get_xy()
            preview = p.pipeline.preview(X, y)
            for step in preview.steps:
                print(f"Step {step['name']}:")
                print(f"  Input: {step['input_shape']}  Output: {step['output_shape']}")
                print(f"  Added: {step['added_columns']}")
                print(f"  Removed: {step['removed_columns']}")
        elif args.subcommand == "apply":
            if not p.pipeline:
                print("No pipeline exists for this project.")
                sys.exit(1)
            p.apply_pipeline(p.pipeline)
            print("Pipeline applied. Data transformed and saved.")

if __name__ == "__main__":
    main()
