# Translated from R (https://CRAN.R-project.org/package=hetGP) to python
# Authors: Binois, M., Gramacy, R., Ludkovski, M

# FILE LICENSE (and for this file only): GNU LPGL-2

import numpy as np


def p1(a, a2, b, b2, t2, theta):
    return (
        (
            2
            * t2
            * (
                63 * t2
                + 9 * 5.0 * np.sqrt(5.0) * b * theta
                - 9 * 5.0 * np.sqrt(5.0) * a * theta
                + 50 * b2
                - 100 * a * b
                + 50 * a2
            )
            * np.exp(2 * np.sqrt(5.0) * a / theta)
            - 63 * t2 * t2
            - 9 * 5.0 * np.sqrt(5.0) * (b + a) * theta * t2
            - 10 * (5 * b2 + 17 * a * b + 5 * a2) * t2
            - 8 * 5.0 * np.sqrt(5.0) * a * b * (b + a) * theta
            - 50 * a2 * b2
        )
        * np.exp(-np.sqrt(5.0) * (b + a) / theta)
        / (36 * np.sqrt(5.0) * theta * t2)
    )


def p3(a, a2, b, b2, t2, theta):
    return (
        (b - a)
        * (
            54 * t2 * t2
            + (54 * np.sqrt(5.0) * b - 54 * np.sqrt(5.0) * a) * theta * t2
            + (105 * b2 - 210 * a * b + 105 * a2) * t2
            + (
                3 * 5.0 * np.sqrt(5.0) * b2 * b
                - 9 * 5.0 * np.sqrt(5.0) * a * b2
                + 9 * 5.0 * np.sqrt(5.0) * a2 * b
                - 3 * 5.0 * np.sqrt(5.0) * a2 * a
            )
            * theta
            + 5 * b2 * b2
            - 20 * a * b2 * b
            + 30 * a2 * b2
            - 20 * a2 * a * b
            + 5 * a2 * a2
        )
        * np.exp(np.sqrt(5.0) * (a - b) / theta)
        / (54 * t2 * t2)
    )


def p4(a, a2, b, t2, theta):
    return (
        -(
            (
                theta
                * (
                    theta
                    * (
                        9 * theta * (7 * theta - 5.0 * np.sqrt(5.0) * (b + a - 2))
                        + 10 * b * (5 * b + 17 * a - 27)
                        + 10 * (5 * a2 - 27 * a + 27)
                    )
                    - 8 * 5.0 * np.sqrt(5.0) * (a - 1) * (b - 1) * (b + a - 2)
                )
                + 50 * (a - 1) * (a - 1) * (b - 2) * b
                + 50 * (a - 1) * (a - 1)
            )
            * np.exp(2 * np.sqrt(5.0) * b / theta)
        )
        * np.exp(-np.sqrt(5.0) * (b - a + 2) / theta)
        / (36 * np.sqrt(5.0) * theta * t2)
    )
