import argparse

from .register_pptx import register_pptx


def mvt_main():
    parser = argparse.ArgumentParser(prog="mvt", description="CLI for the Microscopy Vessel Toolkit")
    commands = parser.add_subparsers(dest="cmd")

    register_pptx_cmd = commands.add_parser("register-pptx", help="Automatically register patches in a PPTX file.")
    register_pptx_cmd.add_argument("path", type=str, help="Path to the PPTX file")
    register_pptx_cmd.add_argument(
        "-o", "--output", type=str, help="Path to the output file. (Default: INPUT/PATH_registered.pptx)", default=None
    )

    args = parser.parse_args()
    match args.cmd:
        case "register-pptx":
            register_pptx(args.path, args.output)
        case _:
            parser.print_help()


if __name__ == "__main__":
    mvt_main()
