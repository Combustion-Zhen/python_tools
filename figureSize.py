# cm inch transfer for matplotlib
def cm2inch(*tupl):
    inch = 2.54
    return tuple(i/inch for i in tupl)

# calculate the subplot size and plot height
# shape: two dimensional list, nrow and ncol
# margin: four dimensional list, left, bottom, right, top
# space: two dimensional list, height and width
def UniformSubplots( plotWidth = 9, 
                     shape = [1, 1], 
                     ratio = 0.8, 
                     margin = [1.5, 1.2, 0.3, 0.3], 
                     space = [1, 1] ): 

    subplotWidth = ( plotWidth 
                    - margin[0] - margin[2] 
                    - ( shape[1] - 1 ) * space[1] 
                   ) / shape[1]

    subplotHeight = subplotWidth * ratio

    plotHeight = ( shape[0] * subplotHeight
                  + margin[1] + margin[3]
                  + ( shape[0] -1 ) * space[0] )

    return plotHeight, subplotHeight, subplotWidth
