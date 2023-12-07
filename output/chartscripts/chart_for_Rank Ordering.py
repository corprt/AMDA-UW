import matplotlib.pyplot as plt
import numpy as np

# Data for vintage 1
min_prob_1 = [32, 870, 908, 924, 935, 942, 947, 951, 955, 958]
max_prob_1 = [869, 907,923 ,934 ,941 ,946 ,950 ,954 ,957 ,964]
events_1 = [870,503 ,346 ,281 ,224 ,138 ,107 ,116 ,71  	42 ]
nonevents_1 = [2415	2852	2846	3267	3225	2885	3048	3462	2825   	 	  	 	  	 	  	 	              	 	      	   	      	   	     	 	             	 	             	 	             	 	             	         	  	           ]
dev_rate_1 = ['26.48%', '14.99%', '10.84%', '7.92%', '6.49%',
             '4.57%', '3.39%', '3.24%', '2.45%','1.36%']

# Data for vintage 2
min_prob_2 = [119  	  	      	   	      	   	     	       	          	       	          	       	          	         ,
             880  	  	      	   	      	   	     	       	          	       	          	       	          	         ,
             912  	  	      	   	      	   	     	       	          	       	          	       	            ,
            927   	         	             	             	             	             ,
            936  	      	    	      	    	     	             ,
            943  	      	    	      	    	     	             ,
            948  	      	    	      	    	     	             ,
           	 	           	              	              	               	         	           	              	               	         	           	              	               	         	           	              	               	         ]
max_prob_2 =[879  	 	    		      	    		         	    		       		         	    		       		         	    		       		         	    		       		        	 		         	    			         	    			         	    			         ] 
events_2 =[141  	 	    		    		    		    	    			    	    			    	    			 		    			    		    	 		    	    	 		    	    	 		    		 	    		 		   		     		 	    		 		   		     		 	    				 		 	    					 		 	    					 		 	    				         ] 
nonevents_2 =[480  	 	    		    		    		    	    			    	    			    	    			 		    			    		    	 		    	    	 		    	    	 		    		 	    		 		   		     		 	    		 		   		     		 	    				 		 	    					 		 	    					 		 	    				         ] 
val_rate_2=['22.71%' ,'15 .69%' ,'8 .79%' ,'8 .32%' ,'5 .45%',
           		          		          		          		          		           		            		           		            		           		            		           		            		           		         ]

# Plotting the data
fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111)
x=np.arange(0,len(min_prob))
width=0.
rects=ax.bar(x-width/4,min_prob,width/4,color='b',label='Minimum Probability')
rects=ax.bar(x,width/4,max_prob,color='g',label='Maximum Probability')
rects=ax.bar(x+width/4,event,width/4,color='r',label='Events')

x=np.arange(0,len(min_prob))
plt.xticks(x,min_prob)

plt.xlabel('Probability Range')
plt.ylabel('Number of Events and Nonevents')
plt.title('Rank Ordering')

ax.legend()

# Saving the plot as a png file in the specified location
plt.savefig('output\chartimages\Rank Ordering.png')