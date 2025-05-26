
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
from torch.nn.functional import kl_div, softmax, log_softmax
from torch.distributions import MultivariateNormal

from scipy.stats import beta

from .mvnorm import multivariate_normal_cdf as Phi


class KLDivLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(KLDivLoss, self).__init__()

        self.temperature = temperature
    def forward(self, emb1, emb2):
        emb1 = softmax(emb1/self.temperature, dim=1).detach()
        emb2 = log_softmax(emb2/self.temperature, dim=1)
        loss_kldiv = kl_div(emb2, emb1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)
        loss_kldiv = torch.mean(loss_kldiv)
        return loss_kldiv


class RankingLoss(nn.Module):
    def __init__(self, neg_penalty=0.03):
        super(RankingLoss, self).__init__()

        self.neg_penalty = neg_penalty
    def forward(self, ranks, labels, class_ids_loaded, device):
        '''
        for each correct it should be higher then the absence 
        '''
        labels = labels[:, class_ids_loaded]
        ranks_loaded = ranks[:, class_ids_loaded]
        neg_labels = 1+(labels*-1)
        loss_rank = torch.zeros(1).to(device)
        for i in range(len(labels)):
            correct = ranks_loaded[i, labels[i]==1]
            wrong = ranks_loaded[i, neg_labels[i]==1]
            correct = correct.reshape((-1, 1)).repeat((1, len(wrong)))
            wrong = wrong.repeat(len(correct)).reshape(len(correct), -1)
            image_level_penalty = ((self.neg_penalty+wrong) - correct)
            image_level_penalty[image_level_penalty<0]=0
            loss_rank += image_level_penalty.sum()
        loss_rank /=len(labels)

        return loss_rank


class CosineLoss(nn.Module):
    
    def forward(self, cxr, ehr):
        a_norm = ehr / ehr.norm(dim=1)[:, None]
        b_norm = cxr / cxr.norm(dim=1)[:, None]
        loss = 1 - torch.mean(torch.diagonal(torch.mm(a_norm, b_norm.t()), 0))
        
        return loss


class CopulaLoss(nn.Module):

    def __init__(self, dim=256, K=3, rho_scale=-5, family="Gumbel"):
        super(CopulaLoss, self).__init__()
        """
        Because Gumbel Copula represents the correlations of the largest signals of both distributions
        """
        self.theta = nn.Parameter(torch.ones(1) * 1)

        self.pi_x = nn.Parameter(torch.ones([K]) / K)
        self.pi_y = nn.Parameter(torch.ones([K]) / K)

        if family == "Gumbel":
            self.copula_cdf = self.gumbel_cdf
            self.copula_pdf = self.gumbel_cdf # TODO: Use the correct pdf later...
        elif family == "Clayton":
            self.copula_cdf = self.clayton_cdf
            self.copula_pdf = self.clayton_pdf
        elif family == "Gaussian":
            self.copula_cdf = self.gaussian_copula_cdf
            self.copula_pdf = self.gaussian_copula_pdf
        elif family == "Frank":
            self.copula_cdf = self.frank_cdf
            self.copula_pdf = self.frank_pdf


        self.mu_x = nn.Parameter(torch.zeros([K, dim]))
        self.mu_y = nn.Parameter(torch.zeros([K, dim]))
        self.log_cov_x = nn.Parameter(torch.ones([K, dim]) * -4)
        self.log_cov_y = nn.Parameter(torch.ones([K, dim]) * -4)
        self.K = K

    def forward(self, x, y):

        pi_x = self.pi_x.log_softmax(dim=-1)
        pi_y = self.pi_y.log_softmax(dim=-1)

        cov_x = torch.log1p(torch.exp(self.log_cov_x)).clamp(min=1e-15)
        cov_y = torch.log1p(torch.exp(self.log_cov_y)).clamp(min=1e-15)

        log_u_list = [self.mvn_cdf(x, self.mu_x[k], cov_x[k]) for k in range(self.K)]
        log_v_list = [self.mvn_cdf(y, self.mu_y[k], cov_y[k]) for k in range(self.K)]
        c_list = [self.copula_pdf(u, v) for u, v in zip(log_u_list, log_v_list)]
        # c = self.gumbel_pdf(u, v)
        # c = self.gumbel_cdf(u, v)

        # loss = torch.cat(c_list)
        u_log_pdf = [self.mvn_pdf(x, self.mu_x[k], cov_x[k]) for k in range(self.K)]
        v_log_pdf = [self.mvn_pdf(y, self.mu_y[k], cov_y[k]) for k in range(self.K)]
        loss = torch.stack(c_list, dim=1) + torch.stack(u_log_pdf, dim=1) + pi_x + torch.stack(v_log_pdf, dim=1) + pi_y
        loss = torch.logsumexp(loss, -1).mean(0)

        assert not torch.isinf(torch.stack(u_log_pdf, dim=1)).any()
        assert not torch.isinf(torch.stack(v_log_pdf, dim=1)).any()
        assert not torch.isinf(torch.stack(c_list, dim=1)).any()

        assert loss.dim() == 0
        assert not torch.isnan(loss)

        return - loss  #  ELBO = negative of likelihood

    def rsample(self, n_samples=[0]):
        """
        Sample (gradient-preserving) from Gaussian mixture
        Only for y
        """
        pi_x = self.pi_x.softmax(dim=-1).clamp(min=1e-15)
        pi_y = self.pi_y.softmax(dim=-1).clamp(min=1e-15)

        cov_x = torch.log1p(torch.exp(self.log_cov_x)).clamp(min=1e-15)
        cov_y = torch.log1p(torch.exp(self.log_cov_y)).clamp(min=1e-15)

        # Using reparameterization tricks
        assert not torch.isnan(self.mu_x).any()
        assert not torch.isnan(cov_x).any()
        assert not torch.isnan(pi_x).any()
        assert not torch.isnan(self.mu_y).any()
        assert not torch.isnan(cov_y).any()
        assert not torch.isnan(pi_y).any()
        x_samples = [MultivariateNormal(self.mu_x[k], scale_tril=torch.diag(cov_x[k])).rsample(sample_shape=n_samples) * pi_x[k] for k in range(self.K)]
        y_samples = [MultivariateNormal(self.mu_y[k], scale_tril=torch.diag(cov_y[k])).rsample(sample_shape=n_samples) * pi_y[k] for k in range(self.K)]

        x_samples = torch.stack(x_samples, dim=0).sum(dim=0)
        y_samples = torch.stack(y_samples, dim=0).sum(dim=0)
        assert not torch.isnan(x_samples).any()
        assert not torch.isnan(y_samples).any()

        assert not torch.isinf(x_samples).any()
        assert not torch.isinf(y_samples).any()

        return y_samples

    def gumbel_cdf(self, log_u, log_v):
        theta = self.theta.clamp(min=1)

        g = (-log_u) ** theta + (-log_v) ** theta
        log_copula_cdf = - g ** (1 / theta)

        return log_copula_cdf

    def gumbel_density(self, u, v):
        """
        Density of Bivariate Gumbel Copula
        """
        g = (- torch.log(u)) ** self.theta + (- torch.log(v)) ** self.theta
        copula_cdf = torch.exp(- g ** (1 / self.theta))
        density = g ** (2 * (1 - self.theta) / self.theta) * ((self.theta - 1) * g ** (- 1 / self.theta) + 1)
        density *= copula_cdf
        density *= (- torch.log(u)) ** (self.theta - 1) * (- torch.log(v)) ** (self.theta - 1)
        density /= u * v

        return density

    def clayton_cdf(self, log_u, log_v):
        alpha = self.theta.clamp(1e-5)

        log_copula_cdf = torch.exp(-alpha * log_u) + torch.exp(-alpha * log_v) - 1
        log_copula_cdf = torch.log(log_copula_cdf.clamp())

        return log_copula_cdf

    def clayton_pdf(self, log_u, log_v):
        pass

    def frank_cdf(self, log_u, log_v):
        theta = self.theta

        u = torch.exp(log_u)
        v = torch.exp(log_v)

        cdf = torch.exp(-theta * u - 1) * torch.exp(-theta * v - 1)
        cdf /= torch.exp(-theta - 1)
        cdf += 1
        cdf = - torch.log(cdf.clamp(1e-9)) / theta
        cdf = torch.log(cdf.clamp(1e-9))

        assert not torch.isnan(cdf).any()

        return cdf


    def frank_pdf(self, log_u, log_v):
        theta = self.theta

        # Performed some standardization
        u = torch.exp(- ((log_u - torch.mean(log_u)) / torch.std(log_u)) ** 2 / 2)
        v = torch.exp(- ((log_v - torch.mean(log_v)) / torch.std(log_v)) ** 2 / 2)

        pdf = (torch.exp(-theta) - 1) * -theta * torch.exp(-theta * (u + v))
        pdf = torch.log(pdf.clamp(1e-5))
        pdf -= 2 * torch.logsumexp(torch.stack([
            -theta * torch.ones_like(u),
            -theta * u,
            -theta * v,
            -theta * (u+v)
        ]), dim=0)

        return pdf

    def gaussian_copula_cdf(self, log_u, log_v):
        pass

    def gaussian_copula_pdf(self, log_u, log_v):
        rho = torch.tanh(self.theta)

        u = torch.exp(- ((log_u - torch.mean(log_u)) / torch.std(log_u)) ** 2 / 2).clamp(min=1e-6, max=0.99999)
        v = torch.exp(- ((log_v - torch.mean(log_v)) / torch.std(log_v)) ** 2 / 2).clamp(min=1e-6, max=0.99999)

        a = np.sqrt(2) * torch.erfinv(2 * u - 1)
        b = np.sqrt(2) * torch.erfinv(2 * v - 1)

        assert not torch.isinf(a).any()
        assert not torch.isinf(b).any()

        log_pdf = - ((a ** 2 + b ** 2) * rho ** 2 - 2 * a * b * rho) / (2 * (1 - rho ** 2))
        log_pdf -= 0.5 * torch.log((1 - rho ** 2).clamp(min=0.001))
        return log_pdf

    def mvn_cdf(self, x, mu, cov):
        """
        Log CDF of multivariate normal distribution
        """
        m = mu - x  # actually do P(Y-value<0)
        m_shape = m.shape
        d = m_shape[-1]
        z = -m / cov
        q = (torch.erfc(-z*0.70710678118654746171500846685)/2)
        q = q.clamp(min=1e-15)
        phi = torch.log(q).sum(-1)
        phi = phi.clamp(max=phi[phi < 0].max(-1)[0])

        return phi
        # return Phi(x, mu, cov)

    def mvn_pdf(self, x, mu, cov):
        """
        PDF of multivariate normal distribution
        """
        log_pdf = MultivariateNormal(mu, scale_tril=torch.diag(cov)).log_prob(x)

        return log_pdf

    def mvn_log_pdf(self, x):
        """
        PDF of multivariate normal distribution
        """

        return (-torch.log(torch.sqrt(2 * torch.pi))
                - torch.log(self.std_dev)
                - ((x - self.mu) ** 2) / (2 * self.std_dev ** 2)).sum(dim=-1)

class Copula3DLoss(nn.Module):

    def __init__(self, dim=256, K=3, rho_scale=-5, family="Gumbel"):
        super(Copula3DLoss, self).__init__()
        """
        Copula 3D Loss
        """
        self.theta = nn.Parameter(torch.ones(1) * 1)

        self.pi_x = nn.Parameter(torch.ones([K]) / K)
        self.pi_y = nn.Parameter(torch.ones([K]) / K)
        self.pi_z = nn.Parameter(torch.ones([K]) / K)

        if family == "Gumbel":
            self.copula_cdf = self.gumbel_cdf
            self.copula_pdf = self.gumbel_cdf # TODO: Use the correct pdf later...
        elif family == "Clayton":
            self.copula_cdf = self.clayton_cdf
            self.copula_pdf = self.clayton_pdf
        elif family == "Gaussian":
            self.copula_cdf = self.gaussian_copula_cdf
            self.copula_pdf = self.gaussian_copula_pdf
        elif family == "Frank":
            self.copula_cdf = self.frank_cdf
            self.copula_pdf = self.frank_pdf


        self.mu_x = nn.Parameter(torch.zeros([K, dim]))
        self.mu_y = nn.Parameter(torch.zeros([K, dim]))
        self.mu_z = nn.Parameter(torch.zeros([K, dim]))
        self.log_cov_x = nn.Parameter(torch.ones([K, dim]) * -4)
        self.log_cov_y = nn.Parameter(torch.ones([K, dim]) * -4)
        self.log_cov_z = nn.Parameter(torch.ones([K, dim]) * -4)
        self.K = K

    def forward(self, x, y, z):

        pi_x = self.pi_x.log_softmax(dim=-1)
        pi_y = self.pi_y.log_softmax(dim=-1)
        pi_z = self.pi_z.log_softmax(dim=-1)

        cov_x = torch.log1p(torch.exp(self.log_cov_x)).clamp(min=1e-15)
        cov_y = torch.log1p(torch.exp(self.log_cov_y)).clamp(min=1e-15)
        cov_z = torch.log1p(torch.exp(self.log_cov_z)).clamp(min=1e-15)

        log_u_list = [self.mvn_cdf(x, self.mu_x[k], cov_x[k]) for k in range(self.K)]
        log_v_list = [self.mvn_cdf(y, self.mu_y[k], cov_y[k]) for k in range(self.K)]
        log_w_list = [self.mvn_cdf(z, self.mu_z[k], cov_z[k]) for k in range(self.K)]
        c_list = [self.copula_pdf(u, v, w) for u, v, w in zip(log_u_list, log_v_list, log_w_list)]
        # c = self.gumbel_pdf(u, v)
        # c = self.gumbel_cdf(u, v)

        # loss = torch.cat(c_list)
        u_log_pdf = [self.mvn_pdf(x, self.mu_x[k], cov_x[k]) for k in range(self.K)]
        v_log_pdf = [self.mvn_pdf(y, self.mu_y[k], cov_y[k]) for k in range(self.K)]
        w_log_pdf = [self.mvn_pdf(z, self.mu_z[k], cov_z[k]) for k in range(self.K)]
        loss = torch.stack(c_list, dim=1) + torch.stack(u_log_pdf, dim=1) + pi_x + torch.stack(v_log_pdf, dim=1) + pi_y + torch.stack(w_log_pdf, dim=1) + pi_z
        loss = torch.logsumexp(loss, -1).mean(0)

        assert not torch.isinf(torch.stack(u_log_pdf, dim=1)).any()
        assert not torch.isinf(torch.stack(v_log_pdf, dim=1)).any()
        assert not torch.isinf(torch.stack(w_log_pdf, dim=1)).any()
        assert not torch.isinf(torch.stack(c_list, dim=1)).any()

        assert loss.dim() == 0
        assert not torch.isnan(loss)

        return - loss  #  ELBO = negative of likelihood

    def rsample(self, n_samples=[0]):
        """
        Sample (gradient-preserving) from Gaussian mixture
        Only for y
        """
        pi_x = self.pi_x.softmax(dim=-1).clamp(min=1e-15)
        pi_y = self.pi_y.softmax(dim=-1).clamp(min=1e-15)

        cov_x = torch.log1p(torch.exp(self.log_cov_x)).clamp(min=1e-15)
        cov_y = torch.log1p(torch.exp(self.log_cov_y)).clamp(min=1e-15)

        # Using reparameterization tricks
        assert not torch.isnan(self.mu_x).any()
        assert not torch.isnan(cov_x).any()
        assert not torch.isnan(pi_x).any()
        assert not torch.isnan(self.mu_y).any()
        assert not torch.isnan(cov_y).any()
        assert not torch.isnan(pi_y).any()
        x_samples = [MultivariateNormal(self.mu_x[k], scale_tril=torch.diag(cov_x[k])).rsample(sample_shape=n_samples) * pi_x[k] for k in range(self.K)]
        y_samples = [MultivariateNormal(self.mu_y[k], scale_tril=torch.diag(cov_y[k])).rsample(sample_shape=n_samples) * pi_y[k] for k in range(self.K)]

        x_samples = torch.stack(x_samples, dim=0).sum(dim=0)
        y_samples = torch.stack(y_samples, dim=0).sum(dim=0)
        assert not torch.isnan(x_samples).any()
        assert not torch.isnan(y_samples).any()

        assert not torch.isinf(x_samples).any()
        assert not torch.isinf(y_samples).any()

        return y_samples

    def gumbel_cdf(self, log_u, log_v, log_w):
        theta = self.theta.clamp(min=1)

        g = (-log_u) ** theta + (-log_v) ** theta + (-log_w) ** theta
        log_copula_cdf = - g ** (1 / theta)

        return log_copula_cdf

    def gumbel_density(self, u, v, w):
        """
        Density of Bivariate Gumbel Copula
        """
        g = (- torch.log(u)) ** self.theta + (- torch.log(v)) ** self.theta
        copula_cdf = torch.exp(- g ** (1 / self.theta))
        density = g ** (2 * (1 - self.theta) / self.theta) * ((self.theta - 1) * g ** (- 1 / self.theta) + 1)
        density *= copula_cdf
        density *= (- torch.log(u)) ** (self.theta - 1) * (- torch.log(v)) ** (self.theta - 1)
        density /= u * v

        return density

    def clayton_cdf(self, log_u, log_v):
        alpha = self.theta.clamp(1e-5)

        log_copula_cdf = torch.exp(-alpha * log_u) + torch.exp(-alpha * log_v) - 1
        log_copula_cdf = torch.log(log_copula_cdf.clamp())

        return log_copula_cdf

    def clayton_pdf(self, log_u, log_v):
        pass

    def frank_cdf(self, log_u, log_v):
        theta = self.theta

        u = torch.exp(log_u)
        v = torch.exp(log_v)

        cdf = torch.exp(-theta * u - 1) * torch.exp(-theta * v - 1)
        cdf /= torch.exp(-theta - 1)
        cdf += 1
        cdf = - torch.log(cdf.clamp(1e-9)) / theta
        cdf = torch.log(cdf.clamp(1e-9))

        assert not torch.isnan(cdf).any()

        return cdf


    def frank_pdf(self, log_u, log_v):
        theta = self.theta

        # Performed some standardization
        u = torch.exp(- ((log_u - torch.mean(log_u)) / torch.std(log_u)) ** 2 / 2)
        v = torch.exp(- ((log_v - torch.mean(log_v)) / torch.std(log_v)) ** 2 / 2)

        pdf = (torch.exp(-theta) - 1) * -theta * torch.exp(-theta * (u + v))
        pdf = torch.log(pdf.clamp(1e-5))
        pdf -= 2 * torch.logsumexp(torch.stack([
            -theta * torch.ones_like(u),
            -theta * u,
            -theta * v,
            -theta * (u+v)
        ]), dim=0)

        return pdf

    def gaussian_copula_cdf(self, log_u, log_v):
        pass

    def gaussian_copula_pdf(self, log_u, log_v):
        rho = torch.tanh(self.theta)

        u = torch.exp(- ((log_u - torch.mean(log_u)) / torch.std(log_u)) ** 2 / 2).clamp(min=1e-6, max=0.99999)
        v = torch.exp(- ((log_v - torch.mean(log_v)) / torch.std(log_v)) ** 2 / 2).clamp(min=1e-6, max=0.99999)

        a = np.sqrt(2) * torch.erfinv(2 * u - 1)
        b = np.sqrt(2) * torch.erfinv(2 * v - 1)

        assert not torch.isinf(a).any()
        assert not torch.isinf(b).any()

        log_pdf = - ((a ** 2 + b ** 2) * rho ** 2 - 2 * a * b * rho) / (2 * (1 - rho ** 2))
        log_pdf -= 0.5 * torch.log((1 - rho ** 2).clamp(min=0.001))
        return log_pdf

    def mvn_cdf(self, x, mu, cov):
        """
        Log CDF of multivariate normal distribution
        """
        m = mu - x  # actually do P(Y-value<0)
        m_shape = m.shape
        d = m_shape[-1]
        z = -m / cov
        q = (torch.erfc(-z*0.70710678118654746171500846685)/2)
        q = q.clamp(min=1e-15)
        phi = torch.log(q).sum(-1)
        phi = phi.clamp(max=phi[phi < 0].max(-1)[0])

        return phi
        # return Phi(x, mu, cov)

    def mvn_pdf(self, x, mu, cov):
        """
        PDF of multivariate normal distribution
        """
        log_pdf = MultivariateNormal(mu, scale_tril=torch.diag(cov)).log_prob(x)

        return log_pdf

    def mvn_log_pdf(self, x):
        """
        PDF of multivariate normal distribution
        """

        return (-torch.log(torch.sqrt(2 * torch.pi))
                - torch.log(self.std_dev)
                - ((x - self.mu) ** 2) / (2 * self.std_dev ** 2)).sum(dim=-1)