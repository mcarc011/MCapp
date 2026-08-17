import matplotlib.pyplot as plt
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
    page_title="VU Lens Graph",
    layout="wide"
)


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


# ============================================================
# SMALL FRAME ORIENTATION DIAGRAM
# ============================================================

def plot_frame_diagram(
    lens,
    angle_deg
):

    Eye, Rx, Z, ZBb, ZF, nCT = lens

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

    theta = np.deg2rad(
        angle_deg
    )

    half_length = (
        frame_half_length(
            Rx,
            angle_deg
        )
    )

    x1 = (
        -half_length
        * np.cos(theta)
    )

    y1 = (
        -half_length
        * np.sin(theta)
    )

    x2 = (
        half_length
        * np.cos(theta)
    )

    y2 = (
        half_length
        * np.sin(theta)
    )

    fig, ax = plt.subplots(
        figsize=(4, 3)
    )

    rectangle_x = [
        -A / 2,
        A / 2,
        A / 2,
        -A / 2,
        -A / 2
    ]

    rectangle_y = [
        -B / 2,
        -B / 2,
        B / 2,
        B / 2,
        -B / 2
    ]

    ax.plot(
        rectangle_x,
        rectangle_y,
        linewidth=2
    )

    ax.plot(
        [
            x1,
            x2
        ],
        [
            y1,
            y2
        ],
        linewidth=3
    )

    ax.scatter(
        [0],
        [0],
        s=25
    )

    margin = 4

    ax.set_xlim(
        -A / 2 - margin,
        A / 2 + margin
    )

    ax.set_ylim(
        -B / 2 - margin,
        B / 2 + margin
    )

    ax.set_aspect(
        "equal"
    )

    eye_name = (
        "Right"
        if Eye == "R"
        else "Left"
    )

    ax.set_title(
        f"{eye_name} frame — "
        f"{angle_deg:.0f}°",
        fontsize=11
    )

    ax.tick_params(
        labelsize=8
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    return fig


# ============================================================
# MAIN LENS CROSS SECTION
# ============================================================

def SliceTk(
    lens,
    angle_deg
):

    Eye, Rx, Z, ZBb, ZF, nCT = lens

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
    # LIMIT TO FRAME BOX / FED
    # ========================================================

    frame_limit = (
        frame_half_length(
            Rx,
            angle_deg
        )
    )

    valid = (
        np.abs(d)
        <= frame_limit
    )

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
        f"A = {A:.2f} mm   |   "
        f"B = {B:.2f} mm   |   "
        f"FED = {FED:.2f} mm   |   "
        f"slice half-length = {frame_limit:.2f} mm"
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

    # Small orientation diagram only
    frame_col, blank_col = st.columns(
        [
            1,
            3
        ]
    )

    with frame_col:

        frame_fig = (
            plot_frame_diagram(
                RightEye,
                angle
            )
        )

        st.pyplot(
            frame_fig,
            use_container_width=True
        )

        plt.close(
            frame_fig
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

    frame_col, blank_col = st.columns(
        [
            1,
            3
        ]
    )

    with frame_col:

        frame_fig = (
            plot_frame_diagram(
                LeftEye,
                angle
            )
        )

        st.pyplot(
            frame_fig,
            use_container_width=True
        )

        plt.close(
            frame_fig
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
