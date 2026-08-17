import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import json
import requests
from Crypto.Cipher import AES
import base64
from numpy import array


# ============================================================
# GLOBAL GRID
# ============================================================

x = np.arange(-42.5, 43.0, 0.5)

Y, X = np.meshgrid(x, x)

R = np.sqrt(X**2 + Y**2)

Theta = np.arctan2(Y, -X + 0.00000001)

ZT = R - R


# ============================================================
# DOWNLOAD JOB
# ============================================================

def downloadjobs(jobname):

    global ZT

    start = str(int(jobname) - 1)
    final = str(int(jobname) + 1)

    username = st.secrets["username"]
    password = st.secrets["password"]
    url = st.secrets["mainurl"]
    key = st.secrets["mckey"]

    aeskey = bytes.fromhex(key)

    auth = (username, password)

    data = {
        "jobsrequest": [start, final]
    }

    response = requests.post(
        url,
        json=data,
        auth=auth
    )

    received = json.loads(
        response.content.decode("utf-8")
    )

    for fname in received:

        tempreceived = {}

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
                received[fname]
                .split("&")[i]
            )

        # --------------------------------------------
        # DECRYPT JOB
        # --------------------------------------------

        cipher = AES.new(
            aeskey,
            AES.MODE_EAX,
            bytes.fromhex(
                tempreceived["nonce"]
            )
        )

        jobdata = cipher.decrypt_and_verify(
            bytes.fromhex(
                tempreceived["data"]
            ),
            bytes.fromhex(
                tempreceived["tag"]
            )
        )

        jobdata = jobdata.decode("utf-8")

        Zr = jobdata.split("@")[0]
        Zl = jobdata.split("@")[1]

        Zr = Zr.split("\n")
        Zl = Zl.split("\n")

        ZTr = ZT - ZT
        ZTl = ZT - ZT

        # --------------------------------------------
        # RECONSTRUCT 3D SURFACES
        # --------------------------------------------

        for i in range(len(ZTr)):

            for j in range(len(ZTl)):

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

        SliceTkR = tempreceived["rslice"]
        SliceTkL = tempreceived["lslice"]

        # Keep your original serialization format
        SliceTkR = eval(SliceTkR)
        SliceTkL = eval(SliceTkL)

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

        return SliceTkR, SliceTkL


# ============================================================
# BILINEAR INTERPOLATION
# ============================================================

def sample_surface(surface, xs, ys):
    """
    Sample a 171x171 surface at arbitrary X,Y positions.

    Uses bilinear interpolation so the rotating slice
    moves smoothly instead of jumping between pixels.
    """

    # Grid:
    # -42.5 -> index 0
    # 0     -> index 85
    # +42.5 -> index 170

    fx = (xs + 42.5) / 0.5
    fy = (ys + 42.5) / 0.5

    # Because of mesh/grid orientation:
    # surface[row_y, column_x]

    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)

    x1 = x0 + 1
    y1 = y0 + 1

    x0 = np.clip(x0, 0, 170)
    x1 = np.clip(x1, 0, 170)

    y0 = np.clip(y0, 0, 170)
    y1 = np.clip(y1, 0, 170)

    dx = fx - x0
    dy = fy - y0

    z00 = surface[y0, x0]
    z10 = surface[y0, x1]
    z01 = surface[y1, x0]
    z11 = surface[y1, x1]

    z0 = (
        z00 * (1 - dx)
        + z10 * dx
    )

    z1 = (
        z01 * (1 - dx)
        + z11 * dx
    )

    z = (
        z0 * (1 - dy)
        + z1 * dy
    )

    return z


# ============================================================
# ROTATING SLICE
# ============================================================

def get_rotated_slice(surface, angle_deg):
    """
    Extract a diameter through the 3D surface
    at angle angle_deg.
    """

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
# FIND FRAME EDGE ALONG CURRENT ANGLE
# ============================================================

def frame_radius(rx, angle_deg):
    """
    Approximate frame shape as ellipse using
    HBOX and VBOX.

    Returns radius from center to frame edge
    along the selected angle.
    """

    A = float(
        rx.get("HBOX", 50)
    )

    B = float(
        rx.get("VBOX", 40)
    )

    a = A / 2.0
    b = B / 2.0

    theta = np.deg2rad(
        angle_deg
    )

    denominator = np.sqrt(
        (
            np.cos(theta) / a
        )**2
        +
        (
            np.sin(theta) / b
        )**2
    )

    if denominator == 0:
        return min(a, b)

    return 1.0 / denominator


# ============================================================
# ROTATING LENS CROSS SECTION
# ============================================================

def SliceTk(lens, angle_deg):

    Eye, Rx, Z, ZBb, ZF, nCT = lens

    # --------------------------------------------
    # PARAMETERS
    # --------------------------------------------

    CT = float(
        Rx.get(
            "BCTHK",
            nCT
        )
    )

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

    # --------------------------------------------
    # ROTATED SURFACE SLICES
    # --------------------------------------------

    d, xs, ys, Z_slice = (
        get_rotated_slice(
            Z,
            angle_deg
        )
    )

    _, _, _, ZF_slice = (
        get_rotated_slice(
            ZF,
            angle_deg
        )
    )

    _, _, _, ZBb_slice = (
        get_rotated_slice(
            ZBb,
            angle_deg
        )
    )

    # --------------------------------------------
    # THICKNESS ALONG THIS SLICE
    # --------------------------------------------

    thickness = (
        nCT
        - ZF_slice
        + Z_slice
    )

    thickness = np.round(
        thickness,
        3
    )

    # --------------------------------------------
    # CALCULATE FRAME EDGE
    # --------------------------------------------

    edge_radius = frame_radius(
        Rx,
        angle_deg
    )

    valid = (
        np.abs(d)
        <= edge_radius
    )

    lens_d = d[valid]

    lens_thickness = (
        thickness[valid]
    )

    Z_slice_valid = (
        Z_slice[valid]
    )

    ZF_slice_valid = (
        ZF_slice[valid]
    )

    ZBb_slice_valid = (
        ZBb_slice[valid]
    )

    # --------------------------------------------
    # STATISTICS
    # --------------------------------------------

    center_index = np.argmin(
        np.abs(lens_d)
    )

    center_thickness = (
        lens_thickness[
            center_index
        ]
    )

    minimum_thickness = (
        np.min(
            lens_thickness
        )
    )

    maximum_thickness = (
        np.max(
            lens_thickness
        )
    )

    left_edge_thickness = (
        lens_thickness[0]
    )

    right_edge_thickness = (
        lens_thickness[-1]
    )

    # ========================================================
    # SURFACE CROSS SECTION
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    # Front surface
    ax.plot(
        lens_d,
        ZF_slice_valid - nCT,
        linewidth=2,
        label="Front Surface"
    )

    # Designed/back surface
    ax.plot(
        lens_d,
        Z_slice_valid,
        linewidth=2,
        label="Designed Surface"
    )

    # Original/base surface
    ax.plot(
        lens_d,
        ZBb_slice_valid
        + (CT - nCT),
        linestyle="--",
        linewidth=1.5,
        label="Base Surface"
    )

    # --------------------------------------------
    # Shade physical thickness
    # --------------------------------------------

    ax.fill_between(
        lens_d,
        ZF_slice_valid - nCT,
        Z_slice_valid,
        alpha=0.15
    )

    # --------------------------------------------
    # CENTER THICKNESS
    # --------------------------------------------

    center_x = lens_d[
        center_index
    ]

    center_front = (
        ZF_slice_valid[
            center_index
        ]
        - nCT
    )

    center_back = (
        Z_slice_valid[
            center_index
        ]
    )

    ax.plot(
        [center_x, center_x],
        [
            center_front,
            center_back
        ],
        linewidth=2
    )

    ax.text(
        center_x,
        (
            center_front
            + center_back
        ) / 2,
        f"  {center_thickness:.3f} mm",
        fontsize=12
    )

    # --------------------------------------------
    # LEFT EDGE
    # --------------------------------------------

    ax.plot(
        [
            lens_d[0],
            lens_d[0]
        ],
        [
            ZF_slice_valid[0]
            - nCT,
            Z_slice_valid[0]
        ],
        linewidth=2
    )

    ax.text(
        lens_d[0],
        (
            (
                ZF_slice_valid[0]
                - nCT
            )
            + Z_slice_valid[0]
        ) / 2,
        f"  {left_edge_thickness:.3f}",
        fontsize=10
    )

    # --------------------------------------------
    # RIGHT EDGE
    # --------------------------------------------

    ax.plot(
        [
            lens_d[-1],
            lens_d[-1]
        ],
        [
            ZF_slice_valid[-1]
            - nCT,
            Z_slice_valid[-1]
        ],
        linewidth=2
    )

    ax.text(
        lens_d[-1],
        (
            (
                ZF_slice_valid[-1]
                - nCT
            )
            + Z_slice_valid[-1]
        ) / 2,
        f"  {right_edge_thickness:.3f}",
        fontsize=10
    )

    # --------------------------------------------
    # GRAPH FORMATTING
    # --------------------------------------------

    eye_name = (
        "Right"
        if Eye == "R"
        else "Left"
    )

    ax.set_title(
        f"{eye_name} Eye — "
        f"{angle_deg:.1f}° Slice"
    )

    ax.set_xlabel(
        "Position Along Slice (mm)"
    )

    ax.set_ylabel(
        "Surface Sag (mm)"
    )

    ax.axvline(
        0,
        linewidth=0.5,
        alpha=0.4
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # ========================================================
    # THICKNESS PROFILE
    # ========================================================

    fig2, ax2 = plt.subplots(
        figsize=(10, 3.5)
    )

    ax2.plot(
        lens_d,
        lens_thickness,
        linewidth=2
    )

    ax2.fill_between(
        lens_d,
        lens_thickness,
        alpha=0.15
    )

    ax2.set_title(
        "Thickness Along Slice"
    )

    ax2.set_xlabel(
        "Position Along Slice (mm)"
    )

    ax2.set_ylabel(
        "Thickness (mm)"
    )

    ax2.axvline(
        0,
        linewidth=0.5,
        alpha=0.4
    )

    ax2.grid(
        alpha=0.2
    )

    fig2.tight_layout()

    st.pyplot(fig2)

    plt.close(fig2)

    # ========================================================
    # THICKNESS NUMBERS
    # ========================================================

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Center",
            f"{center_thickness:.3f} mm"
        )

    with c2:

        st.metric(
            "Minimum",
            f"{minimum_thickness:.3f} mm"
        )

    with c3:

        st.metric(
            "Maximum",
            f"{maximum_thickness:.3f} mm"
        )


# ============================================================
# APP
# ============================================================

st.write(
    "# VU Lens Graph"
)


# ============================================================
# SLICE ANGLE
# ============================================================

angle = st.slider(
    "Slice Angle",
    min_value=0.0,
    max_value=180.0,
    value=0.0,
    step=1.0
)


# Show direction conceptually
st.caption(
    f"Viewing lens cross-section at {angle:.1f}°"
)


# ============================================================
# DEFAULT DATA
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
# LOAD JOB
# ============================================================

if "show" in st.session_state:

    jobv = st.session_state[
        "show"
    ]

    RightEye, LeftEye = (
        downloadjobs(
            jobv
        )
    )

    RxDisplay = {}

    for key in RightEye[1]:

        if key in showdat:

            if key == "CRIB":

                RightEye[1][key] = round(
                    RightEye[1][key],
                    2
                )

                LeftEye[1][key] = round(
                    LeftEye[1][key],
                    2
                )

            RxDisplay[key] = (
                str(
                    RightEye[1][key]
                )
                + ", "
                + str(
                    LeftEye[1][key]
                )
            )

else:

    RxDisplay = {}


# ============================================================
# JOB INPUT
# ============================================================

jobn = st.text_input(
    "Job Number"
)

plot = st.button(
    "Plot"
)


if plot:

    st.session_state[
        "show"
    ] = jobn

    st.rerun()


# ============================================================
# DISPLAY
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "RIGHT",
        "LEFT",
        "PRSC"
    ]
)


with tab1:

    SliceTk(
        RightEye,
        angle
    )


with tab2:

    SliceTk(
        LeftEye,
        angle
    )


with tab3:

    st.write(
        RxDisplay
    )
