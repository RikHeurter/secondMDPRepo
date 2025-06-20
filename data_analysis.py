import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.figure(dpi=600)

mydata = pd.read_csv("data/compressed_data_arrival_39.590000.csv", delimiter=',')

npdata = mydata.to_numpy()
mysum = 0
is_dist = 0
last_data_point = 39.59
for i, item in npdata:
    mysum += i*item
    is_dist += item
    # print(i, item)
print(mysum, is_dist)
my_data = mysum


# Easy plot, simply plotting the average number of jobs waiting.
num_waiting = []
for i in range(40):
     myfile = "data/compressed_data_arrival_" + str(i) + ".csv"
     mydata = pd.read_csv(myfile, delimiter=',')
     npdata = mydata.to_numpy()
     mysum = 0
     is_dist = 0
     for i, item in npdata:
         mysum += i*item
         is_dist += item
         # print(i, item)
     # print(i, mysum, is_dist)
     num_waiting += [mysum]

mydatapoints = np.array(list(range(40)))
# mydatapoints.append(last_data_point)
# num_waiting.append(my_data)

mu = 39.6
rhos = [i/mu for i in mydatapoints[1:]]
num_waiting2 = [rho/(1-rho) for rho in rhos]

plt.plot(mydatapoints/mu, num_waiting, label="Markov Decision Process")
plt.plot(mydatapoints[1:]/mu, num_waiting2, label="Processor Sharing")
plt.xlabel("Average system load",fontstyle = 'italic', fontfamily='sans-serif')
plt.ylabel("Mean number in the system",fontstyle = 'italic', fontfamily='sans-serif')
plt.title('Optimised Markov Decision Process \nversus Optimal Processor Sharing',fontstyle = 'italic', fontfamily='sans-serif')
plt.legend()
plt.savefig("Found_error.png")
plt.show()

plt.figure(dpi=600)

plt.plot(mydatapoints/mu, num_waiting, label="Markov Decision Process")
plt.xlabel("Average system load",fontstyle = 'italic', fontfamily='sans-serif')
plt.ylabel("Mean number in the system",fontstyle = 'italic', fontfamily='sans-serif')
plt.title('Optimised Markov Decision Process \n mean number versus average system load',fontstyle = 'italic', fontfamily='sans-serif')
plt.savefig("data_of_mean_number.png")
plt.show()

# Plot showing the distribution of the times at a specific index.
# Modelled after: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
import numpy as np

indices = list(np.arange(5,40,5))
max_state = 10
#max_state = 30
min_state = 0
states = np.arange(min_state,max_state+1)
indices_props = []

for i in indices:
    myfile = "data/compressed_data_arrival_" + str(i) + ".csv"
    mydata = pd.read_csv(myfile, delimiter=',')
    npdata = mydata.to_numpy()
    index_data = []
    for j, item in npdata:
        if (j < min_state):
            continue
        if (j > max_state):
            break
        index_data.append(item)
    # print(index_data)
    indices_props.append(index_data)

# indices = [1,2]
#myfile = "data/compressed_data_arrival_39.csv"
#mydata = pd.read_csv(myfile, delimiter=',')
#npdata = mydata.to_numpy()
#index_data = []
#for j, item in npdata:
#    if (j < min_state):
#        continue
#    if (j > max_state):
#        break
#    index_data.append(item)
#    # print(index_data)
#indices_props.append(index_data)

#indices = [1,2]
#myfile = "data/compressed_data_arrivalv2_39.csv"
#mydata = pd.read_csv(myfile, delimiter=',')
#npdata = mydata.to_numpy()
#index_data = []
#for j, item in npdata:
#    if (j < min_state):
#        continue
#    if (j > max_state):
#        break
#   index_data.append(item)
#    # print(index_data)
# indices_props.append(index_data)

indices_props = np.array(indices_props)

x = np.arange(len(states))  # the label locations
width = 1/(len(indices)+1)  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained', dpi=600)
# fig.dpi(600)

for index, measurement in zip(indices, indices_props):
    # print(index, len(measurement), measurement)
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label="System load: " + str(round(index/mu, 2)))
    # ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.

states_labels  = ["State " + str(i) for i in states]

ax.set_ylabel('Probability',fontstyle = 'italic', fontfamily='sans-serif')
ax.set_title('Probability of being in states per arrival rate',fontstyle = 'italic', fontfamily='sans-serif')
ax.set_xticks(x + width, states_labels, rotation=45,fontstyle = 'italic', fontfamily='sans-serif')
ax.legend()
plt.savefig("Invariant_distributions.png")

# ax.xlabel("Arrival rate")
# ax.ylabel("Mean number in the system")
# ax.legend(loc='upper left', ncols=5)
# ax.set_ylim(0, 250)

plt.show()

