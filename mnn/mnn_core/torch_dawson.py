# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from math import log, pi, sqrt
from typing import Sequence

import torch
from erfi_pytorch import erfi


EULER_GAMMA = 0.5772156649015329


DAWSON1_INTEGRAL_NEG_COEFFS = (
    (-0.024415375743542616, 0.6441905912698677, 0.0944951677777564, 0.011034916116532503, 0.000931407393639988, 5.4602659863532335e-05, 2.1072678651422e-06, 4.815115134396258e-08, 4.93816935229648e-10),
    (0.09516883204070198, 0.799145074485162, 0.1489106247246331, 0.02276485076363358, 0.002569019985325809, 0.000204206725106341, 1.0786013239031517e-05, 3.394305881862122e-07, 4.8144043951674686e-09),
    (0.2251707771052961, 0.9966593859775033, 0.24154458706917153, 0.05118720409963125, 0.008437064733004108, 0.0010182490054175203, 8.393898228941636e-05, 4.20213360224069e-06, 9.605683810450452e-08),
    (0.31906263086705955, 1.157729809499786, 0.3436464437351062, 0.0994257798497395, 0.025503007119809067, 0.005500279220905265, 0.0009252917467586105, 0.00010617084749033471, 6.096687308294394e-06),
)

DAWSON1_NEG_COEFFS = (
    (0.16999695703523987, 0.2972328375313756, 0.22008847999293366, 0.07285267448885156, 0.06002006719934314, 0.010013831449927596, 0.010909258837995007, 0.0005348533633914443, 0.0009807145963704725),
    (0.6513027823277295, -0.5798374189220877, 0.9381063662835026, -0.4082782128979792, 0.34790282855337873, -0.12321742933540063, 0.06527726660022401, -0.013618873884381362, 0.0042083904337773415),
    (3.785812990298277, -6.327498428580916, 5.361666371737714, -3.239084059541779, 1.8337781098311783, -0.7446375985334174, 0.26396932148206204, -0.057555990324168194, 0.009771027397245355),
    (-14.732069540385776, 26.987232699756944, -18.80916508730925, 10.762121282376945, -4.518376285989208, 1.4395917224474999, -0.27364579561860836, 0.027241015847096185, 0.003296914327413672),
)

DAWSON2_NEG_ASYMPTOTIC_COEFFS = (
    -0.125, 0.3125, -1.0, 4.0625, -20.2265625, 119.80078125,
    -824.671875, 6477.890625, -57223.23486328125,
    561758.9172363281, -6069082.5439453125, 71572017.09594727,
)

DAWSON2_NEG_COEFFS = (
    (0.04456289361497398, -0.0068640113216860005, 0.06968368988514807, -0.005076626910170748, 0.032568377520404625, -0.0022339442474073297, 0.008367331711326493, -0.0004005497899701161, 0.000919750461134635),
    (2.0907639096834174, -3.758768168758032, 3.1080949470616464, -2.0470929425834896, 1.2320768574619576, -0.5578762724344042, 0.22747929263750302, -0.05757045988286828, 0.01289149183504775),
    (107.89416922559883, -197.2704933617773, 150.76405984518064, -95.20446426984915, 49.07719181515287, -19.993470905554904, 6.187146177866233, -1.3067449527328863, 0.15620437384213018),
    (3464.879127245383, -6242.696787133561, 4550.364512459125, -2657.452068978643, 1221.0347194454455, -427.7090486065238, 108.2094406187426, -17.78239522829011, 1.4596430256303885),
)

DAWSON2_INTEGRAL_NEG_COEFFS = (
    (0.027575149379793433, 0.013086762535863928, 0.04110947670661313, 0.005445248525123584, 0.01710726808461111, 0.0006009486402971803, 0.003984895947197359, -3.489122703638879e-05, 0.0004119551894203311),
    (0.5781726818575683, -0.9950292893965544, 0.8596211036220044, -0.5442932972574036, 0.3414710068899002, -0.1496335145787377, 0.063728766036191, -0.015616817635828782, 0.0037467481823447333),
    (20.304620937741618, -37.08302056084885, 28.41840240925292, -17.95503409625239, 9.302330691745976, -3.800152943650162, 1.1878176992400045, -0.2525477577894091, 0.03122009318134736),
    (487.07492704891615, -877.8661547358228, 640.7474385860403, -375.00874585390045, 172.91576028735847, -60.86343789696094, 15.514804843617894, -2.5765862494720415, 0.21620653898362602),
)

DAWSON2_INTEGRAL_POS_COEFFS = (
    (0.19768897016778098, 0.12587144588774266, 0.052444424134850516, -0.06896141287282637, 0.014296526718926566, -0.007332494640939245, 0.005476373992878501, -0.0015781445114326808, 0.00014787008270319263),
    (-0.9296006542130374, 2.1441762232250245, -1.3908478609191357, 0.7472115851493643, -0.34368636516649453, 0.11045612752060398, -0.021901283030162404, 0.0024127555901968995, -0.00011390256114090719),
    (5.8462569337743275, -9.445411423186222, 5.853349318644484, -2.548748418527994, 0.7351576582724473, -0.1371929623327087, 0.015987135575153175, -0.0010622568758324122, 3.085559054228707e-05),
    (-8.653036757781628, 13.92565993721075, -6.600542114177891, 1.9336323169802279, -0.36390706464458294, 0.0443312314522201, -0.0033912129009567435, 0.000148403161720675, -2.8367729164188524e-06),
    (-0.10691735459149024, 1.2358426996782426, -0.9791730133533938, 0.350192115503509, -0.07149016904936, 0.008931854652071294, -0.0006801052962848546, 2.9139951440870103e-05, -5.408449095582141e-07),
    (3.8689532932435027, -4.485397127963087, 1.4055874786324636, -0.2734259214627395, 0.034767111884938584, -0.002910793898093664, 0.00015535219529678424, -4.8077352704222815e-06, 6.583518492788902e-08),
)


def _as_tensor(values: Sequence[Sequence[float]] | Sequence[float], reference: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, dtype=reference.dtype, device=reference.device)


class ChebyshevApproximation:
    @staticmethod
    def evaluate(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.size(-1) == 0:
            return torch.zeros_like(x)
        result = torch.full_like(x, coeffs[0])
        if coeffs.size(-1) == 1:
            return result
        previous = torch.ones_like(x)
        current = x
        result = result + coeffs[1] * current
        for degree in range(2, coeffs.size(-1)):
            previous, current = current, 2 * x * current - previous
            result = result + coeffs[degree] * current
        return result

    @staticmethod
    def evaluate_negative(
        x: torch.Tensor,
        coeffs: torch.Tensor,
        *,
        num_subdivisions: int,
        a: float = 4.0,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        transformed = a / (a + torch.abs(x).pow(alpha))
        return ChebyshevApproximation.evaluate_subdivided(
            transformed,
            coeffs,
            xmin=0.0,
            xmax=1.0,
            num_subdivisions=num_subdivisions,
        )

    @staticmethod
    def evaluate_subdivided(
        x: torch.Tensor,
        coeffs: torch.Tensor,
        *,
        xmin: float,
        xmax: float,
        num_subdivisions: int,
    ) -> torch.Tensor:
        result = torch.zeros_like(x)
        delta = (xmax - xmin) / num_subdivisions
        for index in range(num_subdivisions):
            lower = xmin + delta * index
            upper = xmin + delta * (index + 1)
            mask = (x > lower) & (x <= upper)
            if torch.any(mask):
                result[mask] = ChebyshevApproximation.evaluate(x[mask], coeffs[index])
        return result


@dataclass(frozen=True)
class DawsonFirstOrder:
    chebyshev_divisions: int = 4
    chebyshev_degree: int = 8
    integral_xmin: float = -6.0

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.special.erfcx(-x) * (sqrt(pi) / 2)

    def evaluate_custom(self, x: torch.Tensor) -> torch.Tensor:
        negative_abs = -torch.abs(x)
        asymptotic_region = negative_abs < self.integral_xmin
        positive_region = x > 0
        result = torch.zeros_like(x)
        if torch.any(asymptotic_region):
            result[asymptotic_region] = self.negative_asymptotic(negative_abs[asymptotic_region])
        middle_region = ~asymptotic_region
        if torch.any(middle_region):
            coeffs = _as_tensor(DAWSON1_NEG_COEFFS, x)
            result[middle_region] = ChebyshevApproximation.evaluate_negative(
                negative_abs[middle_region],
                coeffs,
                num_subdivisions=self.chebyshev_divisions,
            )
        if torch.any(positive_region):
            x_pos = x[positive_region]
            result[positive_region] = sqrt(pi) * torch.exp(x_pos * x_pos) - result[positive_region]
        return result

    def negative_asymptotic(self, x: torch.Tensor) -> torch.Tensor:
        term = -0.5 / x
        result = term.clone()
        for n in range(5):
            term = -term * 0.5 * (2 * n + 1) / (x * x)
            result = result + term
        return result

    def integral(self, x: torch.Tensor) -> torch.Tensor:
        negative_abs = -torch.abs(x)
        asymptotic_region = negative_abs < self.integral_xmin
        middle_region = ~asymptotic_region
        positive_region = x > 0
        result = torch.zeros_like(x)
        if torch.any(asymptotic_region):
            result[asymptotic_region] = self.integral_negative_asymptotic(negative_abs[asymptotic_region])
        if torch.any(middle_region):
            coeffs = _as_tensor(DAWSON1_INTEGRAL_NEG_COEFFS, x)
            result[middle_region] = ChebyshevApproximation.evaluate_subdivided(
                negative_abs[middle_region],
                coeffs,
                xmin=self.integral_xmin,
                xmax=0.0,
                num_subdivisions=self.chebyshev_divisions,
            )
        if torch.any(positive_region):
            result[positive_region] = result[positive_region] + (pi / 2) * erfi(x[positive_region])
        return result

    def integral_negative_asymptotic(self, x: torch.Tensor) -> torch.Tensor:
        result = -0.25 * EULER_GAMMA - 0.5 * torch.log(-2 * x)
        for coefficient, power in ((-1 / 8, -2), (3 / 32, -4), (-5 / 32, -6)):
            result = result + coefficient * x.pow(power)
        return result


@dataclass(frozen=True)
class DawsonSecondOrder:
    chebyshev_divisions: int = 4
    chebyshev_degree: int = 8
    positive_integral_divisions: int = 6
    positive_integral_degree: int = 8
    positive_integral_xmax: float = 4.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "_first_order", DawsonFirstOrder())

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        negative_asymptotic_region = x < -10.0
        positive_asymptotic_region = x > 10.0
        middle_region = ~(negative_asymptotic_region | positive_asymptotic_region)
        positive_middle_region = middle_region & (x > 0)
        result = torch.zeros_like(x)
        if torch.any(negative_asymptotic_region):
            result[negative_asymptotic_region] = self.negative_asymptotic(x[negative_asymptotic_region])
        if torch.any(positive_asymptotic_region):
            result[positive_asymptotic_region] = self.positive_asymptotic(x[positive_asymptotic_region])
        if torch.any(middle_region):
            coeffs = _as_tensor(DAWSON2_NEG_COEFFS, x)
            result[middle_region] = ChebyshevApproximation.evaluate_negative(
                -torch.abs(x[middle_region]),
                coeffs,
                num_subdivisions=self.chebyshev_divisions,
            )
        if torch.any(positive_middle_region):
            x_pos = x[positive_middle_region]
            integral_g = self._first_order.integral(-x_pos)
            result[positive_middle_region] = (
                sqrt(pi)
                * torch.exp(x_pos * x_pos)
                * (0.5 * log(2) + 2 * integral_g + (pi / 2) * erfi(x_pos))
                - result[positive_middle_region]
            )
        return result

    def integral(self, x: torch.Tensor) -> torch.Tensor:
        negative_asymptotic_region = x < -10.0
        positive_asymptotic_region = x > self.positive_integral_xmax
        negative_middle_region = ~(negative_asymptotic_region | positive_asymptotic_region) & (x <= 0)
        positive_middle_region = ~(negative_asymptotic_region | positive_asymptotic_region) & (x > 0)
        result = torch.zeros_like(x)
        if torch.any(negative_asymptotic_region):
            result[negative_asymptotic_region] = self.integral_negative_asymptotic(x[negative_asymptotic_region])
        if torch.any(positive_asymptotic_region):
            result[positive_asymptotic_region] = self.integral_positive_asymptotic(x[positive_asymptotic_region])
        if torch.any(negative_middle_region):
            coeffs = _as_tensor(DAWSON2_INTEGRAL_NEG_COEFFS, x)
            result[negative_middle_region] = ChebyshevApproximation.evaluate_negative(
                x[negative_middle_region],
                coeffs,
                num_subdivisions=self.chebyshev_divisions,
            )
        if torch.any(positive_middle_region):
            coeffs = _as_tensor(DAWSON2_INTEGRAL_POS_COEFFS, x)
            x_pos = x[positive_middle_region]
            result[positive_middle_region] = torch.exp(2 * x_pos * x_pos) * ChebyshevApproximation.evaluate_subdivided(
                x_pos,
                coeffs,
                xmin=0.0,
                xmax=self.positive_integral_xmax,
                num_subdivisions=self.positive_integral_divisions,
            )
        return result

    def negative_asymptotic(self, x: torch.Tensor, terms: int = 7) -> torch.Tensor:
        coeffs = _as_tensor(DAWSON2_NEG_ASYMPTOTIC_COEFFS, x)
        result = torch.zeros_like(x)
        for k in range(terms):
            result = result + x.pow(-3 - 2 * k) * coeffs[k]
        return result

    def positive_asymptotic(self, x: torch.Tensor) -> torch.Tensor:
        return (sqrt(pi) / 2) ** 3 * torch.exp(x * x) * torch.erfc(-x).pow(2) * erfi(x)

    def integral_negative_asymptotic(self, x: torch.Tensor, terms: int = 7) -> torch.Tensor:
        coeffs = _as_tensor(DAWSON2_NEG_ASYMPTOTIC_COEFFS, x)
        result = torch.zeros_like(x)
        for k in range(terms):
            power = -2 - 2 * k
            result = result + x.pow(power) * coeffs[k] / power
        return result

    def integral_positive_asymptotic(self, x: torch.Tensor) -> torch.Tensor:
        first = erfi(x)
        second = torch.erfc(-x).pow(2)
        return (pi**2 / 32) * (first - 1) * first * second
