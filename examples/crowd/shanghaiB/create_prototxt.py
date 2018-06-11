def NCLLoss(filename,lambda_,net_num,bottom):
	f1 = open(filename,'w')

	#SliceLayer
	temp="layer {\n "
	temp += 'name:"slicer_conv"\n type:"Slice"\n bottom:"' + bottom + '"\n '
	for idx in xrange(1,net_num+1):
		temp += 'top: "score'+ str(idx) + '"\n '
	temp += 'slice_param {\n  axis: 1\n }\n}\n\n'

	#EltwiseLayer sum
	temp +="layer {\n "
	temp += 'name:"sumscore"\n type:"Eltwise"\n top:"sumscore"\n '
	for idx in xrange(1,net_num+1):
		temp += 'bottom: "score'+ str(idx) + '"\n '
	temp += 'eltwise_param {\n  operation: SUM\n }\n}\n\n'

	#PowerLayer
	temp +="layer {\n "
	temp += 'name:"avgscore"\n type:"Power"\n bottom:"sumscore"\n top:"avgscore"\n '
	temp += 'power_param {\n  power: 1\n  shift: 0\n  scale:'+str(1.0/float(net_num))+'\n }\n include {\n  phase:TEST\n }\n}\n\n'


	#MaeLayer
	temp +="layer {\n "
	temp += 'name:"mae"\n type:"MAELoss"\n bottom:"avgscore"\n bottom:"label"\n top:"mae"\n '
	temp += 'include {\n  phase:TEST\n }\n}\n\n'

	#MaeLayer
	temp +="layer {\n "
	temp += 'name:"mse"\n type:"MSELoss"\n bottom:"avgscore"\n bottom:"label"\n top:"mse"\n '
	temp += 'include {\n  phase:TEST\n }\n}\n\n'

	#NCLLossLayer
	for idx in xrange(1,net_num+1):
		temp +="layer {\n "
		temp += 'name:"loss'+str(idx)+'"\n type:"NCLLoss"\n '	
		temp += 'bottom: "score'+ str(idx) + '"\n bottom: "label"\n bottom: "sumscore"\n top: "loss'+str(idx)+'"\n '
		temp += 'ncl_loss_param {\n  lambda: '+str(lambda_)+'\n  net_num: '+str(net_num)+'\n }\n include {\n  phase: TRAIN\n }\n}\n\n'


        f1.write(temp)
	f1.close()

NCLLoss('nclloss_prototxt',0.0001,128,'conv_score')

	

