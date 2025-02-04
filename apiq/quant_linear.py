import math
import torch
import torch.nn as nn
import torch.nn.functional as F


CLIPMIN = 1e-5

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


class BaseQuantLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_weight_quant = False
        self.use_temporary_parameter = False

    def set_quant_state(self, weight_quant: bool = False):
        self.use_weight_quant = weight_quant


class BinarizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # If x >= 0, return 1. Otherwise, return 0.
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass the gradient directly.
        return grad_output


def binarize_ste(x: torch.Tensor) -> torch.Tensor:
    return BinarizeSTE.apply(x)


class FlexibleDualBinarizer(nn.Module):
    """
    Flexible Dual Binarization module (from DB-LLM)
    """

    def __init__(self, weight: torch.Tensor, group_size: int | None = None):
        """ """
        super().__init__()
        self.group_size = group_size

        # alpha1, alpha2 will be defined after calibration().
        # For simplicity, initialize them as zero-sized parameters here.
        self.alpha1 = nn.Parameter(torch.empty(0))
        self.alpha2 = nn.Parameter(torch.empty(0))

        # Will be set after calibration
        self.num_groups = None
        self.original_shape = None

        # Run calibration to set initial values for alpha1, alpha2
        self.calibration(weight)

    def calibration(self, weight: torch.Tensor):
        """
        Calibrate alpha1, alpha2 using the given weight tensor, possibly in a group-wise manner.
        We reshape the weight so that each row (dimension 0) corresponds to one group.
        Then we compute the local min/max and set:
          alpha1[g] = 2 * s_g,  alpha2[g] = -s_g
        for each group g.

        Args:
            weight (torch.Tensor): The weight matrix (or tensor) to be quantized.
        """
        with torch.no_grad():
            if self.group_size is not None and self.group_size > 0:
                # Flatten all dimensions except the last, then split the last dimension by group_size
                # Example: if weight is [out_features, in_features] and group_size divides in_features,
                # then shape -> (-1, group_size)
                original_size = weight.shape
                last_dim = original_size[-1]
                assert (
                    last_dim % self.group_size == 0
                ), "group_size must evenly divide the last dimension of the weight."

                # Reshape to (num_groups, group_size)
                reshaped = weight.view(-1, self.group_size)
                self.num_groups = reshaped.size(0)
                self.original_shape = original_size

                # Compute scale s for each group
                w_min = reshaped.amin(dim=1, keepdim=True)  # shape (num_groups, 1)
                w_max = reshaped.amax(dim=1, keepdim=True)  # shape (num_groups, 1)
                max_abs = torch.max(w_min.abs(), w_max.abs())  # (num_groups, 1)
                s = max_abs / 2 ** (2 - 1)

                alpha1_init = 2.0 * s  # shape (num_groups, 1)
                alpha2_init = -1.0 * s  # shape (num_groups, 1)

            else:
                # No group-wise approach -> global min/max

                # Compute scale s for the entire weight tensor
                w_min = weight.min()
                w_max = weight.max()
                max_abs = torch.max(w_min.abs(), w_max.abs())
                s = max_abs / 2 ** (2 - 1)

                # We'll keep alpha1, alpha2 as shape (1, 1) for easy broadcasting
                alpha1_init = torch.tensor(
                    [[2.0 * s]], dtype=weight.dtype, device=weight.device
                )
                alpha2_init = torch.tensor(
                    [[-1.0 * s]], dtype=weight.dtype, device=weight.device
                )

                self.num_groups = 1
                self.original_shape = (
                    weight.shape
                )  # Not strictly used but kept for consistency

            # Register them as learnable parameters
            self.alpha1 = nn.Parameter(alpha1_init.squeeze(-1))  # shape (num_groups,)
            self.alpha2 = nn.Parameter(alpha2_init.squeeze(-1))  # shape (num_groups,)

            self.calibrated = True

    def forward(self, weight: torch.Tensor):
        """
        Apply Flexible Dual Binarization to the given weight tensor.
        Requires that calibration() has been called first.

        Args:
            weight (torch.Tensor): The weight matrix (or tensor) to be quantized.

        Returns:
            A pseudo-quantized version of the weight.
        """
        alpha1 = self.alpha1.to(weight.dtype)
        alpha2 = self.alpha2.to(weight.dtype)

        if self.group_size is not None and self.group_size > 0:
            # Group-wise approach
            # Reshape to (num_groups, group_size)
            reshaped = weight.view(-1, self.group_size)
            # Expand alpha1, alpha2 to shape (num_groups, 1)
            alpha1_expanded = alpha1.unsqueeze(1)  # shape (num_groups, 1)
            alpha2_expanded = alpha2.unsqueeze(1)  # shape (num_groups, 1)
            mid = 0.5 * (alpha1_expanded + alpha2_expanded)

            w1_b = binarize_ste(reshaped - mid)
            w2_b = binarize_ste(
                -(reshaped - alpha1_expanded * w1_b - 0.5 * alpha2_expanded)
            )

            w_approx = alpha1_expanded * w1_b + alpha2_expanded * w2_b
            # Reshape back
            w_approx = w_approx.view(self.original_shape)

        else:
            # Global approach (only 1 group)
            alpha1_val = alpha1.view(1, 1)  # shape (1,1)
            alpha2_val = alpha2.view(1, 1)  # shape (1,1)
            mid = 0.5 * (alpha1_val + alpha2_val)

            w1_b = binarize_ste(weight - mid)
            w2_b = binarize_ste(-(weight - alpha1_val * w1_b) - 0.5 * alpha2_val)

            w_approx = alpha1_val * w1_b + alpha2_val * w2_b

        return w_approx

    def register_scales_and_zeros(self):
        pass


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        metric="minmax",
        group_size=None,
        shape=None,
        lwc=False,
        disable_zero_point=False,
    ):
        super().__init__()
        assert 1 <= n_bits <= 16, "bitwidth not supported"
        assert (
            shape[-1] % group_size == 0
        ), "group size should be divisible by the in_feature"
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        self.n_bits = n_bits
        self.group_size = group_size
        self.metric = metric
        self.lwc = lwc

        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        init_value = 4.0  # init value of learnable weight clipping
        if lwc:
            dim1 = (
                int(shape[0] * math.ceil(shape[1] / group_size))
                if group_size
                else shape[0]
            )
            self.upbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)
        self.calibration(x)
        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        return x_dequant

    def fake_quant(self, x, scale, round_zero_point):
        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        # scale_zeros = round_zero_point * scale
        # x_int = round_ste((x + scale_zeros) / scale)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        return x_dequant

    def calibration(self, x):
        if self.group_size:
            x = x.reshape(-1, self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax = x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            xmax = F.sigmoid(self.upbound_factor) * xmax
            xmin = F.sigmoid(self.lowbound_factor) * xmin

        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2 ** (self.n_bits - 1) - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2 ** (self.n_bits - 1) - 1) * torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2**self.n_bits - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -xmin / self.scale
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()

    def register_scales_and_zeros(self):
        self.register_buffer("scales", self.scale)
        self.register_buffer("zeros", self.round_zero_point)
        del self.scale
        del self.round_zero_point


class QuantLinear(BaseQuantLinear):
    def __init__(
        self,
        org_module: nn.Linear,
        quant_method: str = "default",
        weight_quant_params: dict = {},
    ):
        super().__init__()
        self.register_buffer("weight", org_module.weight)
        if org_module.bias is not None:
            self.register_buffer("bias", org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        if quant_method == "default":
            self.weight_quantizer = UniformAffineQuantizer(
                **weight_quant_params, shape=org_module.weight.shape
            )
        elif quant_method == "DB-LLM":
            self.weight_quantizer = FlexibleDualBinarizer(
                weight=org_module.weight, group_size=weight_quant_params["group_size"]
            )
        else:
            raise ValueError(f"Unknown quantization method: {quant_method}")

    def forward(self, x: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
        out = F.linear(x, weight, bias)
        return out


class STEBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        deriv = (input > -1) & (input < 1)
        grad_output = grad_output * deriv
        return grad_output


class BinaryMoSLinear(BaseQuantLinear):

    def __init__(
        self,
        weight,
        bias,
        num_expert,
        do_train,
        zero_point_type=None,
        freeze_original_weights=True,
    ):
        super().__init__()
        # Do not train original weights and biases
        if freeze_original_weights:
            print("Freezing original weights and biases")
            self.register_buffer("weight", weight.data)
            if bias is not None:
                self.register_buffer("bias", bias.data)
            else:
                self.bias = None
        else:
            print("Training original weights and biases")
            self.weight = nn.Parameter(weight.data)
            if bias is not None:
                self.bias = nn.Parameter(bias.data)
            else:
                self.bias = None

        self.out_channel_shape = self.weight.shape[0]
        self.in_channel_shape = self.weight.shape[1]
        self.hidden_dim = self.weight.shape[1]
        self.num_experts = num_expert
        self.do_train = do_train
        self.zero_point_type = zero_point_type

        self.gate_linear = nn.Linear(
            self.hidden_dim, self.num_experts, bias=False, device=self.weight.device
        )
        if self.do_train:
            reduced_rank = 1
            U, S, Vh = torch.linalg.svd(
                abs(weight.data.clone().float()), full_matrices=False
            )
            out_channel_scale = (
                (U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank])))
                .view(-1)
                .repeat(self.num_experts, 1)
            )
            in_channel_scale = (
                (torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh)
                .view(-1)
                .repeat(self.num_experts, 1)
            )
        else:
            in_channel_scale = torch.zeros(self.num_experts, self.weight.shape[1]).to(
                device=self.weight.device
            )
            out_channel_scale = torch.zeros(self.num_experts, self.weight.shape[0]).to(
                device=self.weight.device
            )

        self.register_parameter("in_channel_scale", nn.Parameter(in_channel_scale))
        self.register_parameter("out_channel_scale", nn.Parameter(out_channel_scale))

        # Add output-channel-wise zero point
        if self.zero_point_type == "output_channel":
            print("Using zero point per output channel")
            self.weight_zero_point = nn.Parameter(
                torch.mean(self.weight, dim=1), requires_grad=True
            )
        elif self.zero_point_type == "input_channel":
            print("Using zero point per input channel")
            self.weight_zero_point = nn.Parameter(
                torch.mean(self.weight, dim=0), requires_grad=True
            )
        elif self.zero_point_type is None:
            pass
        else:
            raise ValueError("Invalid zero point type")

    def forward(self, x):

        if self.use_weight_quant:
            *seqlen, hidden_dim = x.shape
            seqlen.append(self.out_channel_shape)
            final_hidden_output_dim = tuple(seqlen)
            x = x.view(-1, hidden_dim)

            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate_linear(x)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

            # we cast back to the input dtype
            routing_weights = routing_weights.to(x.dtype)

            in_scale_expert = routing_weights.matmul(self.in_channel_scale)
            out_scale_expert = routing_weights.matmul(self.out_channel_scale)

            # Binarize the weight
            binary_weight = self.binarize()

            # Subtract the zero point
            if self.zero_point_type == "output_channel":
                quantized_weight = binary_weight - self.weight_zero_point.view(-1, 1)
            elif self.zero_point_type == "input_channel":
                quantized_weight = binary_weight - self.weight_zero_point.view(1, -1)
            else:
                quantized_weight = binary_weight

            if self.bias is not None:
                final_hidden_states = (
                    ((x * in_scale_expert) @ quantized_weight.t()) * out_scale_expert
                ) + self.bias
            else:
                final_hidden_states = (
                    (x * in_scale_expert) @ quantized_weight.t()
                ) * out_scale_expert
            final_hidden_states = final_hidden_states.reshape(final_hidden_output_dim)

            return final_hidden_states
        else:
            return F.linear(x, self.weight, self.bias)

    def binarize(self):
        binary_weight = STEBinary().apply(self.weight)

        return binary_weight

    def extra_repr(self):
        return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, bias={self.bias is not None}, num_experts={self.num_experts}"
