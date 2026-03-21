import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from models.fdta.Temporal_Adapter import History_motion_embedding


class FDTA(nn.Module):
    def __init__(
            self,
            detr: nn.Module,
            detr_framework: str,
            only_detr: bool,
            trajectory_embedding: History_motion_embedding,
            trajectory_modeling: nn.Module,
            id_decoder: nn.Module,
    ):
        super().__init__()
        self.detr = detr
        self.detr_framework = detr_framework
        self.only_detr = only_detr
        self.trajectory_embedding = trajectory_embedding
        self.trajectory_modeling = trajectory_modeling
        self.id_decoder = id_decoder

        if self.id_decoder is not None:
            self.num_id_vocabulary = self.id_decoder.num_id_vocabulary
        else:
            self.num_id_vocabulary = 1000           # hack implementation

        return

    def forward(self, **kwargs):
        assert "part" in kwargs, "Parameter `part` is required for FDTA forward."
        match kwargs["part"]:
            case "detr":
                frames = kwargs["frames"]
                depthmaps = kwargs.get("depthmaps", None)
                if "use_checkpoint" in kwargs:
                    return checkpoint(
                        self.detr, frames, depthmaps,
                        use_reentrant=False,
                    )
                else:
                    return self.detr(samples=frames, depthmaps=depthmaps)
            case "trajectory_embedding":
                seq_info = kwargs["seq_info"]
                return self.trajectory_embedding(seq_info)
            case "trajectory_modeling":
                seq_info = kwargs["seq_info"]
                return self.trajectory_modeling(seq_info)
            case "id_decoder":
                seq_info = kwargs["seq_info"]
                use_decoder_checkpoint = kwargs["use_decoder_checkpoint"] if "use_decoder_checkpoint" in kwargs else False
                return self.id_decoder(seq_info, use_decoder_checkpoint=use_decoder_checkpoint)
            case _:
                raise NotImplementedError(f"FDTA forwarding doesn't support part={kwargs['part']}.")
