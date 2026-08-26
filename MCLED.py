import os
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
import streamlit as st
import json
import requests
from Crypto.Cipher import AES
import base64
from numpy import array


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Vision Unlimited Tools",
    layout="wide"
)

CALLERID_URL='https://vucallerid.streamlit.app'

# ============================================================
# LANDING PAGE / APP ROUTER
# ============================================================

if "vu_page" not in st.session_state:
    st.session_state["vu_page"] = "home"


def get_callerid_url():
    """
    Caller ID can be configured either in .streamlit/secrets.toml:

        callerid_url = "https://YOUR-CALLERID-APP.streamlit.app"

    or with an environment variable:

        CALLERID_URL=https://YOUR-CALLERID-APP.streamlit.app
    """

    try:
        secret_url = st.secrets.get(
            "callerid_url",
            ""
        )
    except Exception:
        secret_url = ""

    return (
        str(secret_url).strip()
        or os.getenv(
            "CALLERID_URL",
            ""
        ).strip()
    )


if st.session_state["vu_page"] == "home":

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1050px;
            padding-top: 4.5rem;
        }

        .vu-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .vu-subtitle {
            text-align: center;
            font-size: 1.1rem;
            opacity: 0.72;
            margin-bottom: 2.6rem;
        }

        .vu-card {
            border: 1px solid rgba(128,128,128,0.28);
            border-radius: 18px;
            padding: 2rem 1.6rem 1.45rem 1.6rem;
            min-height: 195px;
            text-align: center;
            background: rgba(128,128,128,0.035);
            margin-bottom: 0.9rem;
        }

        .vu-card-icon {
            font-size: 2.6rem;
            line-height: 1;
            margin-bottom: 0.75rem;
        }

        .vu-card-title {
            font-size: 1.55rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .vu-card-text {
            font-size: 0.98rem;
            opacity: 0.72;
            line-height: 1.45;
        }
        </style>

        <div class="vu-title">Vision Unlimited</div>
        <div class="vu-subtitle">Select an application</div>
        """,
        unsafe_allow_html=True
    )

    left_space, lens_col, caller_col, right_space = st.columns(
        [0.35, 1, 1, 0.35],
        gap="large"
    )

    with lens_col:

        st.markdown(
            """
            <div class="vu-card">
                <div class="vu-card-icon">◉</div>
                <div class="vu-card-title">Lens Graph</div>
                <div class="vu-card-text">
                    Load a lens job, inspect lens geometry,
                    thickness, prescription data, and rotated
                    cross sections.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Open Lens Graph",
            type="primary",
            use_container_width=True,
            key="open_lens_graph"
        ):
            st.session_state["vu_page"] = "lens_graph"
            st.rerun()

    with caller_col:

        st.markdown(
            """
            <div class="vu-card">
                <div class="vu-card-icon">☎</div>
                <div class="vu-card-title">Caller ID</div>
                <div class="vu-card-text">
                    Open the Vision Unlimited live caller-ID
                    viewer for active and missed calls.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        callerid_url = get_callerid_url()

        if callerid_url:

            st.link_button(
                "Open Caller ID",
                callerid_url,
                type="primary",
                use_container_width=True
            )

        else:

            st.button(
                "Open Caller ID",
                disabled=True,
                use_container_width=True,
                key="callerid_not_configured"
            )

            st.caption(
                "Add `callerid_url` to Streamlit secrets "
                "to enable this link."
            )

    st.stop()


# ============================================================
# LENS GRAPH NAVIGATION
# ============================================================

home_col, _ = st.columns(
    [1, 8]
)

with home_col:

    if st.button(
        "← Home",
        use_container_width=True
    ):
        st.session_state["vu_page"] = "home"
        st.rerun()


# ============================================================
# GLOBAL GRID
# ============================================================

x = np.arange(-42.5, 43.0, 0.5)

Y, X = np.meshgrid(x, x)

R = np.sqrt(X**2 + Y**2)

Theta = np.arctan2(Y, -X + 0.00000001)

ZT = R - R


# ============================================================
# DOWNLOAD JOB DATA
# ============================================================

def decode_frame_payload(parts):
    """Read optional trace geometry while remaining compatible with old jobs."""
    for part in reversed(parts[5:]):
        if not part.startswith("FRAME64="):
            continue
        try:
            encoded = part.split("=", 1)[1]
            raw = base64.b64decode(encoded)
            payload = json.loads(raw.decode("utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}
    return {}


def downloadjobs(jobname):

    global ZT

    start = str(int(jobname) - 1)
    final = str(int(jobname) + 1)

    username = st.secrets["username"]
    password = st.secrets["password"]
    url = st.secrets["mainurl"]
    key = st.secrets["mckey"]

    aeskey = bytes.fromhex(key)

    auth = (
        username,
        password
    )

    data = {
        "jobsrequest": [
            start,
            final
        ]
    }

    response = requests.post(
        url,
        json=data,
        auth=auth
    )

    received = json.loads(
        response.content.decode(
            "utf-8"
        )
    )

    for fname in received:

        tempreceived = {}

        parts = received[
            fname
        ].split("&")

        frame_payload = decode_frame_payload(
            parts
        )

        for i, k in enumerate(
            [
                "data",
                "tag",
                "nonce",
                "rslice",
                "lslice"
            ]
        ):

            tempreceived[k] = (
                parts[i]
            )

        # ====================================================
        # DECRYPT
        # ====================================================

        cipher = AES.new(
            aeskey,
            AES.MODE_EAX,
            bytes.fromhex(
                tempreceived[
                    "nonce"
                ]
            )
        )

        jobdata = (
            cipher.decrypt_and_verify(
                bytes.fromhex(
                    tempreceived[
                        "data"
                    ]
                ),
                bytes.fromhex(
                    tempreceived[
                        "tag"
                    ]
                )
            )
        )

        jobdata = jobdata.decode(
            "utf-8"
        )

        Zr = jobdata.split(
            "@"
        )[0]

        Zl = jobdata.split(
            "@"
        )[1]

        Zr = Zr.split(
            "\n"
        )

        Zl = Zl.split(
            "\n"
        )

        ZTr = ZT - ZT
        ZTl = ZT - ZT

        # ====================================================
        # REBUILD 171 x 171 SURFACES
        # ====================================================

        for i in range(
            len(ZTr)
        ):

            for j in range(
                len(ZTl)
            ):

                ZTr[i, j] = float(
                    Zr[
                        i * 171 + j
                    ].split(",")[-1]
                )

                ZTl[i, j] = float(
                    Zl[
                        i * 171 + j
                    ].split(",")[-1]
                )

        SliceTkR = (
            tempreceived[
                "rslice"
            ]
        )

        SliceTkL = (
            tempreceived[
                "lslice"
            ]
        )

        # Keep compatibility with your existing server format
        SliceTkR = eval(
            SliceTkR
        )

        SliceTkL = eval(
            SliceTkL
        )

        SliceTkR = (
            SliceTkR[:2]
            + (ZTr,)
            + SliceTkR[3:]
        )

        SliceTkL = (
            SliceTkL[:2]
            + (ZTl,)
            + SliceTkL[3:]
        )

        SliceTkR = SliceTkR + (
            frame_payload.get("R", {}),
        )

        SliceTkL = SliceTkL + (
            frame_payload.get("L", {}),
        )

        return (
            SliceTkR,
            SliceTkL
        )


# ============================================================
# SURFACE INTERPOLATION
# ============================================================

def sample_surface(
    surface,
    xs,
    ys
):

    """
    Bilinear interpolation of the 171x171 surface.
    """

    fx = (
        xs + 42.5
    ) / 0.5

    fy = (
        ys + 42.5
    ) / 0.5

    x0 = np.floor(
        fx
    ).astype(int)

    y0 = np.floor(
        fy
    ).astype(int)

    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(
        x0,
        0,
        170
    )

    x1 = np.clip(
        x1,
        0,
        170
    )

    y0 = np.clip(
        y0,
        0,
        170
    )

    y1 = np.clip(
        y1,
        0,
        170
    )

    dx = fx - x0
    dy = fy - y0

    z00 = surface[
        y0,
        x0
    ]

    z10 = surface[
        y0,
        x1
    ]

    z01 = surface[
        y1,
        x0
    ]

    z11 = surface[
        y1,
        x1
    ]

    z0 = (
        z00
        * (1 - dx)
        +
        z10
        * dx
    )

    z1 = (
        z01
        * (1 - dx)
        +
        z11
        * dx
    )

    z = (
        z0
        * (1 - dy)
        +
        z1
        * dy
    )

    return z


# ============================================================
# ROTATED SLICE
# ============================================================

def get_rotated_slice(
    surface,
    angle_deg
):

    theta = np.deg2rad(
        angle_deg
    )

    distance = np.arange(
        -42.5,
        43.0,
        0.5
    )

    xs = (
        distance
        * np.cos(theta)
    )

    ys = (
        distance
        * np.sin(theta)
    )

    values = sample_surface(
        surface,
        xs,
        ys
    )

    return (
        distance,
        xs,
        ys,
        values
    )


# ============================================================
# FRAME BOX INTERSECTION
# ============================================================

def frame_half_length(
    rx,
    angle_deg
):

    """
    Determine where the rotating slice intersects
    the frame boxing rectangle.

    A   = HBOX
    B   = VBOX
    FED = additional maximum radial bound
    """

    A = float(
        rx.get(
            "HBOX",
            50
        )
    )

    B = float(
        rx.get(
            "VBOX",
            40
        )
    )

    FED = float(
        rx.get(
            "FED",
            np.sqrt(
                A**2 + B**2
            )
        )
    )

    theta = np.deg2rad(
        angle_deg
    )

    cos_t = abs(
        np.cos(theta)
    )

    sin_t = abs(
        np.sin(theta)
    )

    if cos_t > 1e-10:

        limit_x = (
            A / 2.0
        ) / cos_t

    else:

        limit_x = np.inf

    if sin_t > 1e-10:

        limit_y = (
            B / 2.0
        ) / sin_t

    else:

        limit_y = np.inf

    limit_fed = (
        FED / 2.0
    )

    return min(
        limit_x,
        limit_y,
        limit_fed
    )


def unpack_lens(lens):
    """Return the original six lens values plus optional trace metadata."""
    Eye, Rx, Z, ZBb, ZF, nCT = lens[:6]
    frame = lens[6] if len(lens) > 6 and isinstance(lens[6], dict) else {}
    return Eye, Rx, Z, ZBb, ZF, nCT, frame


def frame_polygon(frame):
    points = np.asarray(
        frame.get("points_mm", []),
        dtype=float
    )
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
        return None
    return points


def fallback_frame_contains(eye, rx, xs, ys):
    """Original centered HBOX/VBOX/FED boundary for older/fallback jobs."""
    x_offset = float(rx.get("XDEC", 0.0))
    if eye == "L":
        x_offset = -x_offset
    y_offset = -float(rx.get("YDEC", 0.0))

    local_x = np.asarray(xs) - x_offset
    local_y = np.asarray(ys) - y_offset
    local_radius = np.sqrt(local_x**2 + local_y**2)

    A = float(rx.get("HBOX", 50.0))
    B = float(rx.get("VBOX", 40.0))
    FED = float(rx.get("FED", np.sqrt(A**2 + B**2)))

    return (
        (np.abs(local_x) <= A / 2.0)
        & (np.abs(local_y) <= B / 2.0)
        & (local_radius <= FED / 2.0)
    )


def frame_contains(lens, xs, ys):
    Eye, Rx, _, _, _, _, frame = unpack_lens(lens)
    polygon = frame_polygon(frame)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    if polygon is not None:
        query_points = np.column_stack((xs.ravel(), ys.ravel()))
        mask = MplPath(polygon).contains_points(
            query_points,
            radius=1e-9
        )
        return mask.reshape(xs.shape)

    return fallback_frame_contains(Eye, Rx, xs, ys)


def frame_slice_mask(lens, angle_deg, distance):
    theta = np.deg2rad(angle_deg)
    xs = distance * np.cos(theta)
    ys = distance * np.sin(theta)
    return frame_contains(lens, xs, ys)


# ============================================================
# SMALL FRAME ORIENTATION DIAGRAM
# ============================================================

def plot_frame_diagram(
    lens,
    angle_deg
):
    Eye, Rx, _, _, _, _, frame = unpack_lens(lens)
    polygon = frame_polygon(frame)
    theta = np.deg2rad(angle_deg)

    sample_distance = np.arange(-42.5, 42.501, 0.1)
    valid = frame_slice_mask(lens, angle_deg, sample_distance)

    if np.any(valid):
        negative_distance = float(sample_distance[valid][0])
        positive_distance = float(sample_distance[valid][-1])
    else:
        half_length = frame_half_length(Rx, angle_deg)
        negative_distance = -half_length
        positive_distance = half_length

    x1 = negative_distance * np.cos(theta)
    y1 = negative_distance * np.sin(theta)
    x2 = positive_distance * np.cos(theta)
    y2 = positive_distance * np.sin(theta)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    if polygon is not None:
        closed = np.vstack((polygon, polygon[0]))
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            color="#b31b34",
            linewidth=2.2,
            label="Traced frame"
        )
        plot_points = polygon
    else:
        A = float(Rx.get("HBOX", 50.0))
        B = float(Rx.get("VBOX", 40.0))
        x_offset = float(Rx.get("XDEC", 0.0))
        if Eye == "L":
            x_offset = -x_offset
        y_offset = -float(Rx.get("YDEC", 0.0))
        rectangle_x = np.array([-A / 2, A / 2, A / 2, -A / 2, -A / 2]) + x_offset
        rectangle_y = np.array([-B / 2, -B / 2, B / 2, B / 2, -B / 2]) + y_offset
        ax.plot(
            rectangle_x,
            rectangle_y,
            color="#777777",
            linewidth=2,
            linestyle="--",
            label="Box/FED fallback"
        )
        plot_points = np.column_stack((rectangle_x, rectangle_y))

    ax.plot(
        [x1, x2],
        [y1, y2],
        color="#1464a5",
        linewidth=3,
        label=f"{angle_deg:.0f}° slice"
    )

    ax.scatter([0], [0], color="#111111", s=30, zorder=4, label="Lens center")

    center_offset = frame.get("center_offset_mm", [])
    if len(center_offset) == 2:
        ax.scatter(
            [center_offset[0]],
            [center_offset[1]],
            color="#e28c1b",
            marker="+",
            s=90,
            linewidth=2,
            zorder=4,
            label="Frame center"
        )

    all_x = np.concatenate((plot_points[:, 0], np.array([0.0, x1, x2])))
    all_y = np.concatenate((plot_points[:, 1], np.array([0.0, y1, y2])))
    margin = 4.0
    ax.set_xlim(float(np.min(all_x)) - margin, float(np.max(all_x)) + margin)
    ax.set_ylim(float(np.min(all_y)) - margin, float(np.max(all_y)) + margin)
    ax.set_aspect("equal")

    eye_name = "Right" if Eye == "R" else "Left"
    ax.set_title(f"{eye_name} frame — {angle_deg:.0f}°", fontsize=12)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def plot_frame_thickness_map(lens):
    """Plot finished thickness over only the traced frame interior."""
    Eye, Rx, Z, _, ZF, nCT, frame = unpack_lens(lens)
    thickness = np.asarray(nCT - ZF + Z, dtype=float)
    inside_frame = frame_contains(lens, X, Y)

    crib = float(Rx.get("CRIB", 85.0))
    inside_blank = R <= crib / 2.0
    valid = inside_frame & inside_blank & np.isfinite(thickness)

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    stats = {
        "minimum": None,
        "maximum": None,
        "mean": None,
        "minimum_x": None,
        "minimum_y": None,
        "grid_points": int(np.count_nonzero(valid)),
    }

    if np.any(valid):
        display = np.where(valid, thickness, np.nan)
        values = thickness[valid]
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        mean = float(np.mean(values))
        minimum_index = int(np.nanargmin(display))
        min_row, min_col = np.unravel_index(minimum_index, display.shape)
        min_x = float(X[min_row, min_col])
        min_y = float(Y[min_row, min_col])

        stats.update(
            {
                "minimum": minimum,
                "maximum": maximum,
                "mean": mean,
                "minimum_x": min_x,
                "minimum_y": min_y,
            }
        )

        if maximum - minimum < 1e-8:
            image = ax.pcolormesh(
                X,
                Y,
                np.ma.masked_invalid(display),
                shading="auto",
                cmap="viridis"
            )
        else:
            levels = np.linspace(minimum, maximum, 18)
            image = ax.contourf(
                X,
                Y,
                np.ma.masked_invalid(display),
                levels=levels,
                cmap="viridis"
            )

        colorbar = fig.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label("Thickness (mm)")
        ax.scatter(
            [min_x],
            [min_y],
            color="#ffef5a",
            edgecolor="#111111",
            marker="X",
            s=85,
            zorder=5
        )
        ax.annotate(
            f"min {minimum:.3f} mm",
            (min_x, min_y),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold"
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No frame area intersects the lens grid",
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="center"
        )

    polygon = frame_polygon(frame)
    if polygon is not None:
        closed = np.vstack((polygon, polygon[0]))
        ax.plot(closed[:, 0], closed[:, 1], color="#b31b34", linewidth=2.2)
    else:
        A = float(Rx.get("HBOX", 50.0))
        B = float(Rx.get("VBOX", 40.0))
        x_offset = float(Rx.get("XDEC", 0.0))
        if Eye == "L":
            x_offset = -x_offset
        y_offset = -float(Rx.get("YDEC", 0.0))
        ax.plot(
            np.array([-A / 2, A / 2, A / 2, -A / 2, -A / 2]) + x_offset,
            np.array([-B / 2, -B / 2, B / 2, B / 2, -B / 2]) + y_offset,
            color="#777777",
            linestyle="--",
            linewidth=2
        )

    ax.scatter([0], [0], color="white", edgecolor="black", s=35, zorder=5)
    eye_name = "Right" if Eye == "R" else "Left"
    ax.set_title(f"{eye_name} Eye — Thickness Inside Finished Frame")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-42.5, 42.5)
    ax.set_ylim(-42.5, 42.5)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig, stats


def render_frame_overview(lens, angle_deg):
    """Render the real frame outline and its full two-dimensional thickness."""
    _, _, _, _, _, _, frame = unpack_lens(lens)
    frame_col, map_col = st.columns([1, 2])

    with frame_col:
        frame_fig = plot_frame_diagram(lens, angle_deg)
        st.pyplot(frame_fig, use_container_width=True)
        plt.close(frame_fig)

        source = frame.get("source", "HBOX/VBOX/FED fallback")
        point_count = int(frame.get("point_count", 0) or 0)
        st.caption(
            f"Boundary: {source} | frameid: {frame.get('frameid', 'unknown')} | "
            f"trace points: {point_count:,} | "
            f"angles: {frame.get('angle_source', 'not supplied')}"
        )
        if frame.get("warning"):
            st.warning(frame["warning"])

    with map_col:
        map_fig, stats = plot_frame_thickness_map(lens)
        st.pyplot(map_fig, use_container_width=True)
        plt.close(map_fig)

        if stats["minimum"] is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric("Frame Minimum", f"{stats['minimum']:.3f} mm")
            m2.metric("Frame Maximum", f"{stats['maximum']:.3f} mm")
            m3.metric("Frame Mean", f"{stats['mean']:.3f} mm")
            st.caption(
                f"Minimum location: x={stats['minimum_x']:.2f} mm, "
                f"y={stats['minimum_y']:.2f} mm | "
                f"tested grid points: {stats['grid_points']:,}"
            )


# ============================================================
# MAIN LENS CROSS SECTION
# ============================================================

def SliceTk(
    lens,
    angle_deg
):

    Eye, Rx, Z, ZBb, ZF, nCT, frame = unpack_lens(lens)

    A = float(
        Rx.get(
            "HBOX",
            50
        )
    )

    B = float(
        Rx.get(
            "VBOX",
            40
        )
    )

    FED = float(
        Rx.get(
            "FED",
            np.sqrt(
                A**2
                + B**2
            )
        )
    )

    CT = float(
        Rx.get(
            "BCTHK",
            nCT
        )
    )

    # ========================================================
    # ROTATED SURFACE CUTS
    # ========================================================

    (
        d,
        xs,
        ys,
        Z_slice
    ) = get_rotated_slice(
        Z,
        angle_deg
    )

    (
        _,
        _,
        _,
        ZF_slice
    ) = get_rotated_slice(
        ZF,
        angle_deg
    )

    (
        _,
        _,
        _,
        ZBb_slice
    ) = get_rotated_slice(
        ZBb,
        angle_deg
    )

    # ========================================================
    # THICKNESS
    # ========================================================

    thickness = (
        nCT
        - ZF_slice
        + Z_slice
    )

    thickness = np.round(
        thickness,
        3
    )

    # ========================================================
    # LIMIT TO THE ACTUAL TRACED FRAME INTERIOR
    # ========================================================

    valid = frame_contains(
        lens,
        xs,
        ys
    )

    crib = float(Rx.get("CRIB", 85.0))
    valid = valid & (np.sqrt(xs**2 + ys**2) <= crib / 2.0)

    if not np.any(valid):
        st.warning(
            f"The {angle_deg:.1f}° slice does not intersect the usable frame area."
        )
        return

    frame_d = d[
        valid
    ]

    frame_Z = Z_slice[
        valid
    ]

    frame_ZF = ZF_slice[
        valid
    ]

    frame_ZBb = ZBb_slice[
        valid
    ]

    frame_thickness = thickness[
        valid
    ]

    # ========================================================
    # STATISTICS
    # ========================================================

    center_index = np.argmin(
        np.abs(
            frame_d
        )
    )

    center_thickness = float(
        frame_thickness[
            center_index
        ]
    )

    min_index = np.argmin(
        frame_thickness
    )

    max_index = np.argmax(
        frame_thickness
    )

    minimum_thickness = float(
        frame_thickness[
            min_index
        ]
    )

    maximum_thickness = float(
        frame_thickness[
            max_index
        ]
    )

    negative_edge_thickness = float(
        frame_thickness[
            0
        ]
    )

    positive_edge_thickness = float(
        frame_thickness[
            -1
        ]
    )

    eye_name = (
        "Right"
        if Eye == "R"
        else "Left"
    )

    # ========================================================
    # LARGE THICKNESS METRICS
    # ========================================================

    st.markdown(
        "### Thickness"
    )

    c1, c2, c3, c4, c5 = st.columns(
        5
    )

    c1.metric(
        "Center",
        f"{center_thickness:.3f} mm"
    )

    c2.metric(
        "Negative Edge",
        f"{negative_edge_thickness:.3f} mm"
    )

    c3.metric(
        "Positive Edge",
        f"{positive_edge_thickness:.3f} mm"
    )

    c4.metric(
        "Minimum",
        f"{minimum_thickness:.3f} mm"
    )

    c5.metric(
        "Maximum",
        f"{maximum_thickness:.3f} mm"
    )

    # ========================================================
    # LARGE CROSS-SECTION GRAPH
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    front_surface = (
        frame_ZF
        - nCT
    )

    designed_surface = (
        frame_Z
    )

    base_surface = (
        frame_ZBb
        + (
            CT
            - nCT
        )
    )

    # --------------------------------------------------------
    # SURFACES
    # --------------------------------------------------------

    ax.plot(
        frame_d,
        front_surface,
        linewidth=3,
        label="Front Surface"
    )

    ax.plot(
        frame_d,
        designed_surface,
        linewidth=3,
        label="Designed Surface"
    )

    ax.plot(
        frame_d,
        base_surface,
        linestyle="--",
        linewidth=2,
        label="Base Surface"
    )

    # Lens body
    ax.fill_between(
        frame_d,
        front_surface,
        designed_surface,
        alpha=0.15
    )

    # ========================================================
    # CENTER THICKNESS
    # ========================================================

    center_x = (
        frame_d[
            center_index
        ]
    )

    center_front = (
        front_surface[
            center_index
        ]
    )

    center_back = (
        designed_surface[
            center_index
        ]
    )

    ax.plot(
        [
            center_x,
            center_x
        ],
        [
            center_front,
            center_back
        ],
        linewidth=3
    )

    ax.text(
        center_x,
        (
            center_front
            + center_back
        ) / 2,
        f"  {center_thickness:.3f} mm",
        fontsize=17,
        fontweight="bold",
        verticalalignment="center"
    )

    # ========================================================
    # NEGATIVE FRAME EDGE
    # ========================================================

    negative_x = (
        frame_d[
            0
        ]
    )

    ax.plot(
        [
            negative_x,
            negative_x
        ],
        [
            front_surface[0],
            designed_surface[0]
        ],
        linewidth=3
    )

    ax.text(
        negative_x,
        (
            front_surface[0]
            + designed_surface[0]
        ) / 2,
        f"{negative_edge_thickness:.3f} mm  ",
        fontsize=17,
        fontweight="bold",
        horizontalalignment="right",
        verticalalignment="center"
    )

    # ========================================================
    # POSITIVE FRAME EDGE
    # ========================================================

    positive_x = (
        frame_d[
            -1
        ]
    )

    ax.plot(
        [
            positive_x,
            positive_x
        ],
        [
            front_surface[-1],
            designed_surface[-1]
        ],
        linewidth=3
    )

    ax.text(
        positive_x,
        (
            front_surface[-1]
            + designed_surface[-1]
        ) / 2,
        f"  {positive_edge_thickness:.3f} mm",
        fontsize=17,
        fontweight="bold",
        horizontalalignment="left",
        verticalalignment="center"
    )

    # ========================================================
    # MINIMUM THICKNESS MARKER
    # ========================================================

    min_x = (
        frame_d[
            min_index
        ]
    )

    min_front = (
        front_surface[
            min_index
        ]
    )

    min_back = (
        designed_surface[
            min_index
        ]
    )

    ax.plot(
        [
            min_x,
            min_x
        ],
        [
            min_front,
            min_back
        ],
        linewidth=2
    )

    ax.scatter(
        [
            min_x
        ],
        [
            min_back
        ],
        s=70
    )

    # ========================================================
    # FORMATTING
    # ========================================================

    ax.set_title(
        f"{eye_name} Eye — "
        f"{angle_deg:.1f}° Slice",
        fontsize=20,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Position Through Frame (mm)",
        fontsize=15
    )

    ax.set_ylabel(
        "",
        fontsize=15
    )

    ax.tick_params(
        axis="both",
        labelsize=13
    )

    ax.axvline(
        0,
        linewidth=1,
        alpha=0.4
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend(
        fontsize=13,
        loc="best"
    )

    # Add margin so edge thickness labels don't get clipped
    x_margin = max(
        3,
        (
            frame_d[-1]
            - frame_d[0]
        ) * 0.08
    )

    ax.set_xlim(
        frame_d[0]
        - x_margin,
        frame_d[-1]
        + x_margin
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(
        fig
    )

    st.caption(
        f"Boundary: {frame.get('source', 'HBOX/VBOX/FED fallback')}   |   "
        f"negative edge = {frame_d[0]:.2f} mm   |   "
        f"positive edge = {frame_d[-1]:.2f} mm   |   "
        f"frameid = {frame.get('frameid', 'unknown')}"
    )


# ============================================================
# STREAMLIT UI
# ============================================================

st.title(
    "VU Lens Graph"
)


# ============================================================
# JOB NUMBER
# ============================================================

top1, top2 = st.columns(
    [
        4,
        1
    ]
)

with top1:

    jobn = st.text_input(
        "Job Number"
    )

with top2:

    st.write("")
    st.write("")

    plot = st.button(
        "Plot",
        use_container_width=True
    )


if plot:

    if jobn.strip():

        st.session_state[
            "show"
        ] = (
            jobn.strip()
        )

        st.rerun()


# ============================================================
# ANGLE
# ============================================================

angle = st.slider(
    "Slice Angle",
    min_value=0.0,
    max_value=180.0,
    value=0.0,
    step=1.0
)

st.caption(
    f"Current meridian: "
    f"{angle:.1f}°"
)


# ============================================================
# DEFAULT GEOMETRY
# ============================================================

Rx = {}

Rx["HBOX"] = 50
Rx["VBOX"] = 40
Rx["XDEC"] = 0
Rx["YDEC"] = 0
Rx["FED"] = 58
Rx["BCTHK"] = 8


ZT = (
    90
    - np.sqrt(
        90**2
        - R**2
    )
)

RightEye = (
    "R",
    Rx,
    ZT,
    ZT,
    ZT,
    2.5
)

LeftEye = (
    "L",
    Rx,
    ZT,
    ZT,
    ZT,
    2.5
)


# ============================================================
# PARAMETERS TO DISPLAY
# ============================================================

showdat = [
    "IPD",
    "SPH",
    "CYL",
    "AX",
    "ADD",
    "PRVM",
    "BUPC",
    "PRVA",
    "LNAMEN",
    "LNSEN",
    "LNAM",
    "LENT",
    "OZONE",
    "CRIB",
    "SEGHT",
    "HBOX",
    "VBOX",
    "DBL",
    "FED",
    "FEDAX",
    "LIND",
    "FRNT"
]


# ============================================================
# LOAD CURRENT JOB
# ============================================================

if (
    "show"
    in st.session_state
):

    jobv = (
        st.session_state[
            "show"
        ]
    )

    try:

        RightEye, LeftEye = (
            downloadjobs(
                jobv
            )
        )

        RxDisplay = {}

        for key in RightEye[
            1
        ]:

            if key in showdat:

                if (
                    key
                    == "CRIB"
                ):

                    RightEye[
                        1
                    ][key] = round(
                        RightEye[
                            1
                        ][key],
                        2
                    )

                    LeftEye[
                        1
                    ][key] = round(
                        LeftEye[
                            1
                        ][key],
                        2
                    )

                RxDisplay[
                    key
                ] = (
                    str(
                        RightEye[
                            1
                        ][key]
                    )
                    + ", "
                    + str(
                        LeftEye[
                            1
                        ][key]
                    )
                )

        st.success(
            f"Job {jobv} loaded"
        )

    except Exception as e:

        st.error(
            f"Could not load job: {e}"
        )

        RxDisplay = {}

else:

    RxDisplay = {}


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "RIGHT",
        "LEFT",
        "PRSC"
    ]
)


# ============================================================
# RIGHT EYE
# ============================================================

with tab1:

    st.subheader(
        "Right Eye"
    )

    render_frame_overview(
        RightEye,
        angle
    )

    # Full-width important graph
    SliceTk(
        RightEye,
        angle
    )


# ============================================================
# LEFT EYE
# ============================================================

with tab2:

    st.subheader(
        "Left Eye"
    )

    render_frame_overview(
        LeftEye,
        angle
    )

    # Full-width important graph
    SliceTk(
        LeftEye,
        angle
    )


# ============================================================
# PRESCRIPTION
# ============================================================

with tab3:

    st.subheader(
        "Job Parameters"
    )

    if RxDisplay:

        pcol1, pcol2 = st.columns(
            2
        )

        items = list(
            RxDisplay.items()
        )

        midpoint = (
            len(items) + 1
        ) // 2

        with pcol1:

            for key, value in (
                items[
                    :midpoint
                ]
            ):

                st.write(
                    f"**{key}:** "
                    f"{value}"
                )

        with pcol2:

            for key, value in (
                items[
                    midpoint:
                ]
            ):

                st.write(
                    f"**{key}:** "
                    f"{value}"
                )

    else:

        st.write(
            "Load a job to view "
            "prescription parameters."
        )
