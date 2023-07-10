from collections.abc import Container  # noqa
from datetime import datetime

from pptx import Presentation
from pptx.shapes.group import GroupShape
from pptx.shapes.picture import Picture

from microscopy_vessels_toolkit.preprocess.stitch import MultiPatchRegistration, PatchStitching


def register_pptx(pptx_path: str, output_path: str = None):
    register = MultiPatchRegistration()
    prs = Presentation(pptx_path)

    if output_path is None:
        pptx_name, pptx_ext = pptx_path.rsplit(".", 1)
        output_path = pptx_name + "_registered." + pptx_ext

    for i, slide in enumerate(prs.slides):
        stitch = PatchStitching.from_pptx_slide(slide)
        if len(stitch) < 2:
            print(f"> Slide {i+1}: No patches found.")
            continue
        else:
            print(f"> Slide {i+1}: {len(stitch)} patches found.")

        t = datetime.now()
        stitch.crop_to_content()
        reg_stitch, reg_debug = register(stitch, return_debug=True, verbose=True)
        relative_offset = {stitch[k].alias: v for k, v in reg_debug["relative_offset"].items()}
        update_pictures_recursive(slide.shapes, relative_offset)
        print(f"\t... registered in {(datetime.now() - t).total_seconds():.1f} seconds.")

    print(f"> Saving output to {output_path}...")
    t = datetime.now()
    prs.save(output_path)
    print(f"\t... saved in {(datetime.now() - t).total_seconds():.1f} seconds.")


def update_pictures_recursive(group, relative_offset):
    for s in group:
        if isinstance(s, GroupShape):
            if s.shapes:
                # Find pictures in group
                update_pictures_recursive(s.shapes, relative_offset)
        elif isinstance(s, Picture) and s.name in relative_offset:
            s.left += int(relative_offset[s.name].x * s.width)
            s.top += int(relative_offset[s.name].y * s.height)
