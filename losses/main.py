import torch.nn as nn
import losses.SmoothPrec_at_K as SmoothPrec_at_K
import losses.Another_P_at_K as Another_P_at_K
import losses.Lifted as Lifted
import losses.Contrastive as Contrastive
import losses.Angular as Angular
import losses.Softmax as Softmax
import losses.LSoftmax as LSoftmax
import losses.ASoftmax as ASoftmax
import losses.ArcFace as ArcFace
import losses.ProxyNCA as ProxyNCA
import losses.Clustering as Clustering
import losses.Npair as NPair
import losses.LMCL as LMCL
import losses.Triplet as Triplet
import losses.DistanceWeighted as DistanceWeighted

def get_loss(n_input=None,k=None,tau=None,n_pos=None,input_dim=None, output_dim=None, margin=None, batch_size=None,method=None):
#metric loss
	sprarsity=0
	err_pos=0
	if method==0:
		print("Using Prec@k loss:")
		loss = SmoothPrec_at_K.SmoothPrec_at_K(n_input=n_input, k=k, tau=tau, margin=margin,batch_size=batch_size,n_pos=n_pos)
	if method==1:
		print("Using ProxyNCA loss")	
		loss = ProxyNCA.ProxyNCA(sz_embed=input_dim, nb_classes=output_dim)
	if method==2:
		print("Using Clustering loss")	
		loss = Clustering.Clustering(margin=margin)
	if method==3:
		print("Using NPair loss")	
		loss = NPair.NPair(margin=margin)		
	if method==4:
		print("Using Angular loss")	
		loss = Angular.Angular(margin=margin)				
	if method==5:
		print("Using Lifted loss")
		loss = Lifted.Lifted(margin=margin)
	if method==6:
		print("Using Triplet loss")
		loss = Triplet.Triplet(margin=margin)
	if method==7:
		print("Using Contrastive loss")	
		loss = Contrastive.Contrastive(margin=margin)
	if method==8:
		print("Using AnotherPrec@k loss:")
		loss = Another_P_at_K.Another_P_at_K(n_input=n_input, k=k, tau=tau, margin=margin,n_pos=n_pos)
	if method==9:
		print("Using DistanceWeighted loss:")
		loss = DistanceWeighted.DistanceWeighted(margin=margin, nb_classes=output_dim)
	
	
#softmax loss	
	if method==10:
		print("Using ArcFace loss")	#ArcFace
		loss = ArcFace.ArcFace(input_dim=input_dim, output_dim=output_dim,margin=margin)
	if method==11:
		print("Using LMCL loss")	#CosFace
		loss = LMCL.LMCL(input_dim=input_dim, output_dim=output_dim,margin=margin)
	if method==12:
		print("Using A-Softmax loss")	#SphereFace
		loss = ASoftmax.ASoftmax(input_dim=input_dim, output_dim=output_dim,margin=margin)	
	if method==13:
		print("Using L-Softmax loss")	#Large Margin
		loss = LSoftmax.LSoftmax(input_dim=input_dim, output_dim=output_dim,margin=margin)					
	if method==14:
		print("Using Common-Softmax loss")	
		loss = Softmax.Softmax(input_dim=input_dim, output_dim=output_dim,margin=margin)
	return loss
