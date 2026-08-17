import base64
import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from Crypto.Cipher import AES


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VU Lens Viewer",
    page_icon="👓",
    layout="wide"
)

st.title("VU Lens Design Viewer")


# ============================================================
# GLOBAL GEOMETRY GRID
# ============================================================

GRID_SIZE = 171
HALF_SIZE = 42.5
STEP = 0.5
CENTER = GRID_SIZE // 2

x = np.arange(-HALF_SIZE, HALF_SIZE + STEP, STEP)

Y, X = np.meshgrid(x, x)

R = np.sqrt(X**2 + Y**2)

# Default surface used before a real job is loaded
ZT_DEFAULT = 90 - np.sqrt(np.maximum(90**2 - R**2, 0))


# ============================================================
# SERVER / DOWNLOAD
# ============================================================

def download_jobs(jobname):
    """
    Download and decrypt right/left lens data from the VU server.
    """

    start = str(int(jobname) - 1)
    final = str(int(jobname) + 1)

    username = st.secrets["username"]
    password = st.secrets["password"]
    url = st.secrets["mainurl"]
    key = st.secrets["mckey"]

    aes_key = bytes.fromhex(key)

    data = {
        "jobsrequest": [start, final]
    }

    try:
        response = requests.post(
            url,
            json=data,
            auth=(username, password),
            timeout=30
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not connect to lens server: {exc}"
        ) from exc

    try:
        received = response.json()

    except Exception as exc:
        raise RuntimeError(
            "Server response was not valid JSON."
        ) from exc

    if not received:
        raise RuntimeError(
            f"No job data returned for job {jobname}."
        )

    # The old code returns the first matching file/job.
    for fname, encrypted_string in received.items():

        pieces = encrypted_string.split("&")

        if len(pieces) < 5:
            raise RuntimeError(
                f"Invalid encrypted job format for {fname}."
            )

        temp_received = {
            "data": pieces[0],
            "tag": pieces[1],
            "nonce": pieces[2],
            "rslice": pieces[3],
            "lslice": pieces[4],
        }

        # ----------------------------------------------------
        # DECRYPT SURFACE DATA
        # ----------------------------------------------------

        cipher = AES.new(
            aes_key,
            AES.MODE_EAX,
            nonce=bytes.fromhex(temp_received["nonce"])
        )

        try:
            jobdata = cipher.decrypt_and_verify(
                bytes.fromhex(temp_received["data"]),
                bytes.fromhex(temp_received["tag"])
            )

        except Exception as exc:
            raise RuntimeError(
                "Lens data could not be decrypted or authenticated."
            ) from exc

        jobdata = jobdata.decode("utf-8")

        try:
            zr_text, zl_text = jobdata.split("@", 1)

        except ValueError as exc:
            raise RuntimeError(
                "Decrypted job data did not contain both lens surfaces."
            ) from exc

        zr_lines = zr_text.splitlines()
        zl_lines = zl_text.splitlines()

        # ----------------------------------------------------
        # REBUILD 171x171 ARRAYS
        # ----------------------------------------------------

        ZTr = parse_surface_lines(zr_lines)
        ZTl = parse_surface_lines(zl_lines)

        # ----------------------------------------------------
        # PARSE THE EXISTING LENS TUPLES
        #
        # literal_eval is much safer than eval().
        # ----------------------------------------------------

        try:
            # Existing VU lens format contains NumPy array(...) expressions.
            # Restrict eval so it only has access to NumPy's array constructor.
            safe_globals = {
                "__builtins__": {},
                "array": np.array,
            }
        
            right_lens = eval(
                temp_received["rslice"],
                safe_globals,
                {}
            )
        
            left_lens = eval(
                temp_received["lslice"],
                safe_globals,
                {}
            )
        
        except Exception as exc:
            raise RuntimeError(
                f"Could not parse right/left lens metadata: {exc}"
            ) from exc

        right_lens = list(right_lens)
        left_lens = list(left_lens)

        # Surface Z is tuple position 2 in your existing format.
        right_lens[2] = ZTr
        left_lens[2] = ZTl

        return tuple(right_lens), tuple(left_lens)

    raise RuntimeError(
        f"Could not find usable data for job {jobname}."
    )


def parse_surface_lines(lines):
    """
    Convert the server's flattened 171x171 surface into a NumPy array.
    """

    expected_points = GRID_SIZE * GRID_SIZE

    if len(lines) < expected_points:
        raise RuntimeError(
            f"Surface contained {len(lines)} points; "
            f"expected {expected_points}."
        )

    values = []

    for line in lines[:expected_points]:

        try:
            values.append(
                float(line.split(",")[-1])
            )

        except Exception as exc:
            raise RuntimeError(
                f"Could not parse surface point: {line}"
            ) from exc

    return np.array(
        values,
        dtype=float
    ).reshape(GRID_SIZE, GRID_SIZE)


# ============================================================
# LENS HELPERS
# ============================================================

def unpack_lens(lens):
    """
    Existing tuple structure:

    Eye, Rx, Z, ZBb, ZF, nCT
    """

    eye, rx, z, z_back_base, z_front, nct = lens

    return {
        "eye": eye,
        "rx": rx,
        "Z": np.array(z, dtype=float),
        "ZBb": np.array(z_back_base, dtype=float),
        "ZF": np.array(z_front, dtype=float),
        "nCT": float(nct),
    }


def calculate_thickness(lens):
    """
    Thickness calculation from the original code.
    """

    data = unpack_lens(lens)

    thickness = (
        data["nCT"]
        - data["ZF"]
        + data["Z"]
    )

    return thickness


def get_thickness_stats(lens):
    thickness = calculate_thickness(lens)

    finite = thickness[np.isfinite(thickness)]

    return {
        "center": float(thickness[CENTER, CENTER]),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


# ============================================================
# CROSS SECTION
# ============================================================

def plot_slice(lens):
    """
    Updated version of your original SliceTk function.
    """

    data = unpack_lens(lens)

    eye = data["eye"]
    rx = data["rx"]
    Z = data["Z"]
    ZBb = data["ZBb"]
    ZF = data["ZF"]
    nCT = data["nCT"]

    A = float(rx.get("HBOX", 50))
    B = float(rx.get("VBOX", 40))
    XDEC = float(rx.get("XDEC", 0))
    YDEC = float(rx.get("YDEC", 0))

    CT = float(rx.get("BCTHK", nCT))

    thickness = np.round(
        calculate_thickness(lens),
        3
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    # --------------------------------------------------------
    # HORIZONTAL SECTION
    # --------------------------------------------------------

    ax.plot(
        x,
        ZBb[CENTER] + (CT - nCT) + HALF_SIZE,
        label="Back Base"
    )

    ax.plot(
        x,
        Z[CENTER] + HALF_SIZE,
        label="Designed Surface"
    )

    ax.plot(
        x,
        ZF[CENTER] - nCT + HALF_SIZE,
        label="Front Surface"
    )

    # --------------------------------------------------------
    # Lens dimensions
    # --------------------------------------------------------

    XP = x / 2.0

    if eye == "R":

        x1 = A / 2.0 - XDEC
        x2 = -A / 2.0 - XDEC

    else:

        x1 = A / 2.0 + XDEC
        x2 = -A / 2.0 + XDEC

    ax.plot(
        np.full_like(x, x1),
        XP - HALF_SIZE
    )

    ax.plot(
        np.full_like(x, x2),
        XP - HALF_SIZE
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    text_locations = [
        (
            -B / 2.0 + YDEC,
            60,
            safe_thickness_value(
                thickness,
                CENTER,
                int(CENTER - B + 2 * YDEC)
            )
        ),
        (
            B / 2.0 + YDEC,
            60,
            safe_thickness_value(
                thickness,
                CENTER,
                int(CENTER + B + 2 * YDEC)
            )
        ),
        (
            0,
            30,
            thickness[CENTER, CENTER]
        )
    ]

    if eye == "R":

        text_locations.extend([
            (
                A / 2.0 - XDEC,
                -60,
                safe_thickness_value(
                    thickness,
                    int(CENTER + A - 2 * XDEC),
                    CENTER
                )
            ),
            (
                -A / 2.0 - XDEC,
                -60,
                safe_thickness_value(
                    thickness,
                    int(CENTER - A - 2 * XDEC),
                    CENTER
                )
            )
        ])

    else:

        text_locations.extend([
            (
                A / 2.0 + XDEC,
                -60,
                safe_thickness_value(
                    thickness,
                    int(CENTER + A + 2 * XDEC),
                    CENTER
                )
            ),
            (
                -A / 2.0 + XDEC,
                -60,
                safe_thickness_value(
                    thickness,
                    int(CENTER - A + 2 * XDEC),
                    CENTER
                )
            )
        ])

    for tx, ty, value in text_locations:

        ax.text(
            tx,
            ty,
            f"{value:.3f}",
            fontsize=13
        )

    # --------------------------------------------------------
    # VERTICAL SECTION
    # --------------------------------------------------------

    ZBb_T = ZBb.T
    Z_T = Z.T
    ZF_T = ZF.T

    ax.plot(
        -x,
        ZBb_T[CENTER] + (CT - nCT) - HALF_SIZE
    )

    ax.plot(
        x,
        Z_T[CENTER] - HALF_SIZE
    )

    ax.plot(
        -x,
        ZF_T[CENTER] - nCT - HALF_SIZE
    )

    ax.plot(
        np.full_like(x, B / 2.0 + YDEC),
        XP + HALF_SIZE
    )

    ax.plot(
        np.full_like(x, -B / 2.0 + YDEC),
        XP + HALF_SIZE
    )

    ax.set_title(
        f"{eye} Eye Cross Section"
    )

    ax.axis("equal")
    ax.axis("off")

    fig.tight_layout()

    return fig


def safe_thickness_value(array, i, j):

    i = int(
        np.clip(i, 0, GRID_SIZE - 1)
    )

    j = int(
        np.clip(j, 0, GRID_SIZE - 1)
    )

    return float(array[i, j])


# ============================================================
# THICKNESS MAP
# ============================================================

def plot_thickness_map(lens):

    data = unpack_lens(lens)

    eye = data["eye"]

    thickness = calculate_thickness(lens)

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    image = ax.imshow(
        thickness,
        extent=[
            -HALF_SIZE,
            HALF_SIZE,
            -HALF_SIZE,
            HALF_SIZE
        ],
        origin="lower",
        aspect="equal"
    )

    ax.set_title(
        f"{eye} Eye Thickness Map"
    )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    fig.colorbar(
        image,
        ax=ax,
        label="Thickness"
    )

    fig.tight_layout()

    return fig


# ============================================================
# 3D SURFACE
# ============================================================

def plot_3d_surface(
    surface,
    title,
    stride=4
):

    fig = plt.figure(
        figsize=(9, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.plot_surface(
        X,
        Y,
        surface,
        rstride=stride,
        cstride=stride,
        linewidth=0
    )

    ax.set_title(title)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Sag")

    fig.tight_layout()

    return fig


# ============================================================
# PARAMETER TABLE
# ============================================================

SHOW_DATA = [
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


def create_parameter_table(
    right_lens,
    left_lens
):

    right_rx = unpack_lens(
        right_lens
    )["rx"]

    left_rx = unpack_lens(
        left_lens
    )["rx"]

    rows = []

    for key in SHOW_DATA:

        if (
            key not in right_rx
            and key not in left_rx
        ):
            continue

        right_value = right_rx.get(
            key,
            ""
        )

        left_value = left_rx.get(
            key,
            ""
        )

        if key == "CRIB":

            try:
                right_value = round(
                    float(right_value),
                    2
                )

            except Exception:
                pass

            try:
                left_value = round(
                    float(left_value),
                    2
                )

            except Exception:
                pass

        rows.append({
            "Parameter": key,
            "Right": right_value,
            "Left": left_value
        })

    return pd.DataFrame(rows)


# ============================================================
# DEFAULT LENS DATA
# ============================================================

default_rx = {
    "HBOX": 50,
    "VBOX": 40,
    "XDEC": 0,
    "YDEC": 0,
    "FED": 58,
    "BCTHK": 8
}

default_right = (
    "R",
    default_rx.copy(),
    ZT_DEFAULT.copy(),
    ZT_DEFAULT.copy(),
    ZT_DEFAULT.copy(),
    2.5
)

default_left = (
    "L",
    default_rx.copy(),
    ZT_DEFAULT.copy(),
    ZT_DEFAULT.copy(),
    ZT_DEFAULT.copy(),
    2.5
)


# ============================================================
# JOB INPUT
# ============================================================

with st.form("job_loader"):

    col1, col2 = st.columns(
        [4, 1]
    )

    with col1:

        job_number = st.text_input(
            "Job Number",
            value=st.session_state.get(
                "job_number",
                ""
            )
        )

    with col2:

        st.write("")
        st.write("")

        submitted = (
            st.form_submit_button(
                "Load Job",
                use_container_width=True
            )
        )


if submitted:

    if not job_number.strip():

        st.warning(
            "Enter a job number."
        )

    else:

        try:

            with st.spinner(
                f"Loading job {job_number}..."
            ):

                right_eye, left_eye = (
                    download_jobs(
                        job_number.strip()
                    )
                )

            st.session_state[
                "right_eye"
            ] = right_eye

            st.session_state[
                "left_eye"
            ] = left_eye

            st.session_state[
                "job_number"
            ] = job_number.strip()

            st.success(
                f"Loaded job {job_number}."
            )

        except Exception as exc:

            st.error(
                f"Could not load job: {exc}"
            )


# ============================================================
# CURRENT JOB
# ============================================================

right_eye = st.session_state.get(
    "right_eye",
    default_right
)

left_eye = st.session_state.get(
    "left_eye",
    default_left
)


loaded_job = st.session_state.get(
    "job_number"
)


if loaded_job:

    st.caption(
        f"Current Job: {loaded_job}"
    )

else:

    st.caption(
        "Displaying default demonstration geometry."
    )


# ============================================================
# TABS
# ============================================================

tab_overview, tab_right, tab_left, tab_thickness, tab_3d = (
    st.tabs([
        "Overview",
        "Right Eye",
        "Left Eye",
        "Thickness",
        "3D"
    ])
)


# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:

    right_stats = (
        get_thickness_stats(
            right_eye
        )
    )

    left_stats = (
        get_thickness_stats(
            left_eye
        )
    )

    st.subheader(
        "Derived Geometry"
    )

    right_header, left_header = (
        st.columns(2)
    )

    with right_header:

        st.markdown(
            "### Right Eye"
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Center Thickness",
            f"{right_stats['center']:.3f}"
        )

        c2.metric(
            "Minimum Thickness",
            f"{right_stats['minimum']:.3f}"
        )

        c3, c4 = st.columns(2)

        c3.metric(
            "Maximum Thickness",
            f"{right_stats['maximum']:.3f}"
        )

        c4.metric(
            "Mean Thickness",
            f"{right_stats['mean']:.3f}"
        )


    with left_header:

        st.markdown(
            "### Left Eye"
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Center Thickness",
            f"{left_stats['center']:.3f}"
        )

        c2.metric(
            "Minimum Thickness",
            f"{left_stats['minimum']:.3f}"
        )

        c3, c4 = st.columns(2)

        c3.metric(
            "Maximum Thickness",
            f"{left_stats['maximum']:.3f}"
        )

        c4.metric(
            "Mean Thickness",
            f"{left_stats['mean']:.3f}"
        )

    st.divider()

    st.subheader(
        "Job Parameters"
    )

    parameter_table = (
        create_parameter_table(
            right_eye,
            left_eye
        )
    )

    st.dataframe(
        parameter_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RIGHT EYE
# ============================================================

with tab_right:

    st.subheader(
        "Right Eye Cross Section"
    )

    fig = plot_slice(
        right_eye
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# LEFT EYE
# ============================================================

with tab_left:

    st.subheader(
        "Left Eye Cross Section"
    )

    fig = plot_slice(
        left_eye
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


# ============================================================
# THICKNESS
# ============================================================

with tab_thickness:

    col_r, col_l = st.columns(2)

    with col_r:

        fig = plot_thickness_map(
            right_eye
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    with col_l:

        fig = plot_thickness_map(
            left_eye
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


# ============================================================
# 3D
# ============================================================

with tab_3d:

    st.subheader(
        "3D Lens Geometry"
    )

    surface_choice = st.selectbox(
        "Surface",
        [
            "Designed Surface",
            "Front Surface",
            "Back Base",
            "Thickness"
        ]
    )

    col_r, col_l = st.columns(2)

    right_data = unpack_lens(
        right_eye
    )

    left_data = unpack_lens(
        left_eye
    )

    if surface_choice == "Designed Surface":

        right_surface = (
            right_data["Z"]
        )

        left_surface = (
            left_data["Z"]
        )

    elif surface_choice == "Front Surface":

        right_surface = (
            right_data["ZF"]
        )

        left_surface = (
            left_data["ZF"]
        )

    elif surface_choice == "Back Base":

        right_surface = (
            right_data["ZBb"]
        )

        left_surface = (
            left_data["ZBb"]
        )

    else:

        right_surface = (
            calculate_thickness(
                right_eye
            )
        )

        left_surface = (
            calculate_thickness(
                left_eye
            )
        )

    with col_r:

        fig = plot_3d_surface(
            right_surface,
            f"Right Eye — {surface_choice}"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    with col_l:

        fig = plot_3d_surface(
            left_surface,
            f"Left Eye — {surface_choice}"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)
