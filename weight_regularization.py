import torch


class Regularization(torch.nn.Module):

    def __init__(self, model, weight_decay, normalization_decay, p=2):

        super(Regularization, self).__init__()

        self.model = model
        self.weight_decay = weight_decay
        self.normalization_decay = normalization_decay
        self.p = p


    def to(self, device):

        self.device = device
        super().to(device)
        return self


    def forward(self, model):

        self.weight_list, self.normalization_list = self.get_weight(model)
        weight_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        normalization_loss = self.regularization_loss(self.normalization_list, self.normalization_decay, p=self.p)

        return weight_loss, normalization_loss


    def get_weight(self, model):

        weight_list = []
        normalization_list = []

        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                if param.dim() == 1:
                    normalization_list.append(weight)
                else:
                    weight_list.append(weight)

        return weight_list, normalization_list


    def regularization_loss(self, weight_list, weight_decay, p=2):

        reg_loss = 0

        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss

        return reg_loss
