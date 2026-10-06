"""The 3D path: camera, renderer, back-projection, framing, labelling.

Single source. scripts/view_backprojection.py is the interactive viewer over
this package and scripts/ingest_render3d.py is the batch driver over it;
neither keeps geometry of its own. Anything added to the 3D path belongs in a
module here, so both inherit it.
"""

from .camera import (Camera, VIEW_TO_EXTRINSIC, camera_ke, extrinsic_of,
                     orbit_extrinsic, pose_angles)
from .config import DEFAULTS, EAR3D_DIR, ROOT
from .backproject import (backproject, backproject_snapped, cliff_map,
                          snap_chain, snap_to_cliff, visible)
from .draw import (ray_lines, save_agreement_render, spheres, strip_lines,
                   write_overlay)
from .frames import (find_ears, frame_from, head_pose, in_frame, load_head,
                     plane_normal)
from .label import (agreement, label_two_pass, landmark_whole_frame,
                    multiview, report_agreement)
from .render import (LINE_MATERIAL, MATERIAL, gui_depth_to_view, light_scene,
                     render)
from .scheme import (CHIN, FOREHEAD, NASION, NOSE_TIP, STRIPS, STRIP_COLOURS,
                     STRIP_NAMES, TRAGION_L, TRAGION_R)
