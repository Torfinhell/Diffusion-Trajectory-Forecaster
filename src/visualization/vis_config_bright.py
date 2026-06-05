canvas_config = {"width": 8, "background_color": "xkcd:white", "tick_on": False}

road_line_config = {
    "Unknown": {
        "color": "xkcd:light grey",
        "linewidth": 1.5,
        "linestyle": "dotted",
        "alpha": 0,
    },
    "BrokenSingleWhite": {
        "color": "xkcd:medium grey",
        "linewidth": 2,
        "linestyle": "--",
        "alpha": 0.5,
    },
    "SolidSingleWhite": {
        "color": "xkcd:medium grey",
        "linewidth": 2,
        "linestyle": "-",
        "alpha": 0.5,
    },
    "SolidDoubleWhite": {
        "color": "xkcd:medium grey",
        "linewidth": 3.5,
        "linestyle": "-",
        "alpha": 0.5,
    },
    "BrokenSingleYellow": {
        "color": "xkcd:yellowish orange",
        "linewidth": 2,
        "linestyle": "--",
        "alpha": 0.5,
    },
    "BrokenDoubleYellow": {
        "color": "xkcd:yellowish orange",
        "linewidth": 3.5,
        "linestyle": "--",
        "alpha": 0.5,
    },
    "SolidSingleYellow": {
        "color": "xkcd:yellowish orange",
        "linewidth": 2,
        "linestyle": "-",
        "alpha": 0.5,
    },
    "SolidDoubleYellow": {
        "color": "xkcd:yellowish orange",
        "linewidth": 2,
        "linestyle": "-",
        "alpha": 0.5,
    },
    "PassingDoubleYellow": {
        "color": "xkcd:yellowish orange",
        "linewidth": 3.5,
        "linestyle": "-.",
        "alpha": 0.5,
    },
}

road_edge_config = {
    "Unknown": {
        "color": "xkcd:brown",
        "linewidth": 2,
        "linestyle": "dotted",
        "alpha": 0.8,
    },
    "Boundary": {
        "color": "xkcd:charcoal",
        "linewidth": 2,
        "linestyle": "-",
        "alpha": 0.8,
    },
    "Median": {"color": "xkcd:sage", "linewidth": 2, "linestyle": "-", "alpha": 0.8},
}

speed_bump_config = {
    "facecolor": "xkcd:sunflower yellow",
    "edgecolor": "xkcd:black",
    "alpha": 1,
}

crosswalk_config = {
    "facecolor": "None",
    "edgecolor": "xkcd:bluish grey",
    "alpha": 0.2,
}

stop_sign_config = {
    "facecolor": "xkcd:red",
    "edgecolor": "none",
    "linewidth": 1.5,
    "radius": 1.5,
    "alpha": 1,
}
