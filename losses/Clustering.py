import torch
import torch.nn as nn
from sklearn import metrics


def Clustering(margin):
    #https://github.com/CongWeilin/cluster-loss-tensorflow/blob/master/metric_learning/metric_loss_ops.py
    return ClusteringLoss(margin=margin) 
class ClusteringLoss(nn.Module):

    def __init__(self,margin):
        super(ClusteringLoss, self).__init__()
        self.margin = margin

    def compute_clustering_score(self,labels, predictions):#nmi score
        return metrics.normalized_mutual_info_score(labels,predictions)

    def pairwise_distance(self, feature):
        #output[i, j] = || feature[i, :] - feature[j, :] ||_2
        dist_matrix = torch.zeros(feature.size()[0],feature.size()[0]).cuda()
        for i in range(feature.size()[0]):
            for j in range(feature.size()[0]):
                dist_matrix[i][j]=torch.nn.functional.pairwise_distance(feature[i].view(1,-1), feature[j].view(1,-1), p=2, eps=1e-6)
        return dist_matrix
    def compute_facility_energy(self,pairwise_distances, centroid_ids):
        """Compute the average travel distance to the assigned centroid.
        eqn4:F(X,S;theta)
        """
        batch_size=pairwise_distances.size()[0]
        centroid_ids_len=centroid_ids.size()[0]
        return -torch.gather(pairwise_distances,1,centroid_ids.repeat(batch_size,1).cuda()).min(dim=1)[0].sum()

    def get_cluster_assignment(self,pairwise_distances, centroid_ids):
        #y_fixed: 1-D tensor of cluster assignment.y=g(S)
        
        batch_size = pairwise_distances.size()[0]
        centroid_ids_len=centroid_ids.size()[0]
        predictions = torch.gather(pairwise_distances,1,centroid_ids.repeat(batch_size,1).cuda()).min(dim=1)[1].float()
        # Deal with numerical instability
        mask = torch.zeros(batch_size).cuda().float()
        for i in range(centroid_ids_len):
            mask[centroid_ids[i]]=1
        a=torch.zeros(batch_size,centroid_ids_len).float().cuda()
        for i in range(centroid_ids_len):
            a[centroid_ids[i]][i]=1
        b=torch.tensor([i for i in range(centroid_ids_len)]).repeat(batch_size,1).float().cuda()
        constraint_one_hot = torch.mul(a,b)
        constraint_vect = constraint_one_hot.t().sum(dim=0)
        y_fixed = (1-mask).mul(predictions)+constraint_vect

        return y_fixed

    def compute_gt_cluster_score(self,pairwise_distances, labels):
        #Compute ground truth facility location score.
        unique_class_ids = torch.unique(labels)
        num_classes = unique_class_ids.size()[0]
        gt_cluster_score = 0
        for i in range(num_classes):
            mask=torch.eq(labels,unique_class_ids[i])
            this_cluster_ids = torch.zeros(mask.sum()).cuda()
            j=0
            for i in range(mask.size()[0]):
                if mask[i]==1:
                    this_cluster_ids[j]=i
                    j+=1
            a=torch.gather(pairwise_distances,1, this_cluster_ids.expand(pairwise_distances.size()[0],this_cluster_ids.size()[0]).long())
            b=torch.gather(a.t(),1, this_cluster_ids.expand(a.size()[1],this_cluster_ids.size()[0]).long())
            pairwise_distances_subset=b.t()
            this_cluster_score=-1.0*pairwise_distances_subset.sum(dim=0).min()[0]
            gt_cluster_score+=this_cluster_score
        return gt_cluster_score

    def update_1d_tensor(self,y, index, value):
        """Updates 1d tensor y so that y[index] = value.
        Args:
        y: 1-D Tensor.
        index: index of y to modify.
        value: new value to write at y[index].
        Returns:
        y_mod: 1-D Tensor. Tensor y after the update.
        """
        value = array_ops.squeeze(value)
        # modify the 1D tensor x at index with value.
        # ex) chosen_ids = update_1D_tensor(chosen_ids, cluster_idx, best_medoid)
        y_before = array_ops.slice(y, [0], [index])
        y_after = array_ops.slice(y, [index + 1], [-1])
        y_mod = array_ops.concat([y_before, [value], y_after], 0)
        return y_mod

    def update_medoid_per_cluster(self,pairwise_distances, pairwise_distances_subset,
                              labels, chosen_ids, cluster_member_ids,
                              cluster_idx, margin_multiplier):
        """Updates the cluster medoid per cluster.
        Args:
        pairwise_distances: 2-D Tensor of pairwise distances.
        pairwise_distances_subset: 2-D Tensor of pairwise distances for one cluster.
        labels: 1-D Tensor of ground truth cluster assignment.
        chosen_ids: 1-D Tensor of cluster centroid indices.
        cluster_member_ids: 1-D Tensor of cluster member indices for one cluster.
        cluster_idx: Index of this one cluster.
        margin_multiplier: multiplication constant.
        margin_type: Type of structured margin to use. Default is nmi.
        Returns:
        chosen_ids: Updated 1-D Tensor of cluster centroid indices.
        """

        def func_cond(iteration, scores_margin):
            del scores_margin  # Unused variable scores_margin.
            return iteration < num_candidates

        def func_body(iteration, scores_margin):
            # swap the current medoid with the candidate cluster member
            candidate_medoid = math_ops.to_int32(cluster_member_ids[iteration])
            tmp_chosen_ids = self.update_1d_tensor(chosen_ids, cluster_idx, candidate_medoid)
            predictions = self.get_cluster_assignment(pairwise_distances, tmp_chosen_ids)
            metric_score = self.compute_clustering_score(labels, predictions, margin_type)
            pad_before = array_ops.zeros([iteration])
            pad_after = array_ops.zeros([num_candidates - 1 - iteration])
            return iteration + 1, scores_margin + array_ops.concat(
                [pad_before, [1.0 - metric_score], pad_after], 0)

        # pairwise_distances_subset is of size [p, 1, 1, p],
        #   the intermediate dummy dimensions at
        #   [1, 2] makes this code work in the edge case where p=1.
        #   this happens if the cluster size is one.
        scores_fac = -1.0 * math_ops.reduce_sum(
          array_ops.squeeze(pairwise_distances_subset, [1, 2]), axis=0)

        iteration = array_ops.constant(0)
        num_candidates = array_ops.size(cluster_member_ids)
        scores_margin = array_ops.zeros([num_candidates])

        _, scores_margin = control_flow_ops.while_loop(func_cond, func_body,
                                                     [iteration, scores_margin])
        candidate_scores = math_ops.add(scores_fac, margin_multiplier * scores_margin)

        argmax_index = math_ops.to_int32(
          math_ops.argmax(candidate_scores, dimension=0))

        best_medoid = math_ops.to_int32(cluster_member_ids[argmax_index])
        chosen_ids = update_1d_tensor(chosen_ids, cluster_idx, best_medoid)
        return chosen_ids

    def update_all_medoids(self,pairwise_distances, predictions, labels, chosen_ids,
                           margin_multiplier, margin_type):
        """Updates all cluster medoids a cluster at a time.
        Args:
        pairwise_distances: 2-D Tensor of pairwise distances.
        predictions: 1-D Tensor of predicted cluster assignment.
        labels: 1-D Tensor of ground truth cluster assignment.
        chosen_ids: 1-D Tensor of cluster centroid indices.
        margin_multiplier: multiplication constant.
        margin_type: Type of structured margin to use. Default is nmi.
        Returns:
        chosen_ids: Updated 1-D Tensor of cluster centroid indices.
        """
        unique_class_ids = torch.unique(labels)
        num_classes = unique_class_ids.size()[0]
        for i in range(num_classes):
            mask = math_ops.equal(
            math_ops.to_int64(predictions), math_ops.to_int64(i))
            this_cluster_ids = array_ops.where(mask)

            pairwise_distances_subset = array_ops.transpose(
            array_ops.gather(
            array_ops.transpose(
            array_ops.gather(pairwise_distances, this_cluster_ids)),
            this_cluster_ids))

            chosen_ids = self.update_medoid_per_cluster(pairwise_distances,
                       pairwise_distances_subset, labels,
                       chosen_ids, this_cluster_ids,
                       i, margin_multiplier)
        return chosen_ids

    def _find_loss_augmented_facility_idx(self,pairwise_distances, labels, chosen_ids,
                                      candidate_ids, margin_multiplier):
        """Find the next centroid that maximizes the loss augmented inference.
        This function is a subroutine called from compute_augmented_facility_locations
        Args:
        pairwise_distances: 2-D Tensor of pairwise distances.
        labels: 1-D Tensor of ground truth cluster assignment.
        chosen_ids: 1-D Tensor of current centroid indices.
        candidate_ids: 1-D Tensor of candidate indices.
        margin_multiplier: multiplication constant.
        margin_type: Type of structured margin to use. Default is nmi.
        Returns:
        integer index.
        """
        num_candidates = candidate_ids.size()[0]
        pairwise_distances_candidate = torch.gather(pairwise_distances,1, candidate_ids.expand(pairwise_distances.size()[0],candidate_ids.size()[0]).long()).t()
        
        if chosen_ids.size()[0]==0:
            candidate_scores=torch.cat([pairwise_distances_candidate.reshape(1,-1)],dim=0).min(dim=0)[0].reshape(num_candidates,-1).sum(dim=1)
        else:
            pairwise_distances_chosen = torch.gather(pairwise_distances,1, chosen_ids.expand(pairwise_distances.size()[0],chosen_ids.size()[0]).long()).t()            
            pairwise_distances_chosen_tile = pairwise_distances_chosen.repeat(1,num_candidates)

            candidate_scores=torch.cat([pairwise_distances_chosen_tile,pairwise_distances_candidate.reshape(1,-1)],dim=0).min(dim=0)[0].reshape(num_candidates,-1).sum(dim=1)


        nmi_scores = torch.zeros(num_candidates).cuda()
        for i in range(num_candidates):
            if chosen_ids.size()[0]==0:
                predictions = self.get_cluster_assignment(
                    pairwise_distances,torch.cat([candidate_ids[i].long().view(1,-1)], 1))
            else:
                predictions = self.get_cluster_assignment(
                    pairwise_distances,torch.cat([chosen_ids.view(1,-1), candidate_ids[i].long().view(1,-1)], 1))
            nmi_score_i = self.compute_clustering_score(labels, predictions)
            pad_before = torch.zeros(i).cuda()
            pad_after = torch.zeros(num_candidates - 1 - i).cuda()
            nmi_score_i = torch.tensor([1.0 - nmi_score_i]).view(1).cuda()
            nmi_scores+=torch.cat([pad_before, nmi_score_i, pad_after])

        candidate_scores = candidate_scores + margin_multiplier * nmi_scores
        argmax_index = torch.max(candidate_scores,dim=0)[1]

        return candidate_ids[argmax_index]

    def compute_augmented_facility_locations(self,pairwise_distances, labels, all_ids, margin_multiplier):
        """
        Computes the centroid locations.
            Args:
            pairwise_distances: 2-D Tensor of pairwise distances.
            labels: 1-D Tensor of ground truth cluster assignment.
            all_ids: 1-D Tensor of all data indices.
            margin_multiplier: multiplication constant.
            Returns:
            chosen_ids: 1-D Tensor of chosen centroid indices.
        """
        num_classes = torch.unique(labels).size()[0]
        chosen_ids=torch.tensor([]).cuda()
        for i in range(num_classes):
            candidate_ids = torch.tensor([]).cuda().float()
            for j in range(all_ids.size()[0]):
                if all_ids[j] not in chosen_ids:
                    if candidate_ids.size()[0]==0:
                        candidate_ids=all_ids[j].view(-1).cuda().float()
                    else:
                        candidate_ids=torch.cat([candidate_ids.view(-1),all_ids[j].float().view(-1)],0)
            new_chosen_idx = self._find_loss_augmented_facility_idx(pairwise_distances,labels, chosen_ids,candidate_ids,margin_multiplier)
            chosen_ids = torch.cat([chosen_ids.view(-1).float(), new_chosen_idx.view(-1).float()], 0).long()
        return chosen_ids

    def compute_augmented_facility_locations_pam(self,pairwise_distances,labels,margin_multiplier,chosen_ids,pam_max_iter=5):
        """Refine the cluster centroids with PAM local search.
        For fixed iterations, alternate between updating the cluster assignment
        and updating cluster medoids.
        Args:
        pairwise_distances: 2-D Tensor of pairwise distances.
        labels: 1-D Tensor of ground truth cluster assignment.
        margin_multiplier: multiplication constant.
        margin_type: Type of structured margin to use. Default is nmi.
        chosen_ids: 1-D Tensor of initial estimate of cluster centroids.
        pam_max_iter: Number of refinement iterations.
        Returns:
        chosen_ids: Updated 1-D Tensor of cluster centroid indices.
        """
        for _ in range(pam_max_iter):
            # update the cluster assignment given the chosen_ids (S_pred)
            predictions = self.get_cluster_assignment(pairwise_distances, chosen_ids)

            # update the medoids per each cluster
            chosen_ids = self.update_all_medoids(pairwise_distances, predictions, labels,
                    chosen_ids, margin_multiplier)

        return chosen_ids

    def forward(self, embedding, label):
        pairwise_distances = self.pairwise_distance(embedding)
        all_ids = torch.arange(0,label.size()[0]).cuda()

        # S wait Compute the loss augmented inference and get the cluster centroids.
        chosen_ids = self.compute_augmented_facility_locations(pairwise_distances, label,all_ids, self.margin)
        print(chosen_ids)
        # F(X,S;theta) OK Given the predicted centroids, compute the clustering score.
        score_pred = self.compute_facility_energy(pairwise_distances, chosen_ids)

        #chosen_ids = self.compute_augmented_facility_locations_pam(pairwise_distances,labels,margin_multiplier,
        #                                                  margin_type,chosen_ids)
        #score_pred = self.compute_facility_energy(pairwise_distances, chosen_ids)

        # y OK Given the predicted centroids, compute the cluster assignments.
        predictions = self.get_cluster_assignment(pairwise_distances, chosen_ids)

        # NMI OK Compute the clustering (NMI) score between the two assignments.
        clustering_score_pred = self.compute_clustering_score(label, predictions)

        # F~ OK Compute the clustering score from labels.   label->y*
        score_gt = self.compute_gt_cluster_score(pairwise_distances, label)

        clustering_loss = torch.max(score_pred + self.margin * (1.0 - clustering_score_pred) - score_gt,0)[0]
        return clustering_loss,0,0