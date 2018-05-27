import matplotlib.image as mpimg    # mpimg: read img
import matplotlib.pyplot as plt     # plt: show img
import numpy as np

def normPdf(x, mu, sigma):
    return(np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))

def distance(a, b):
    sum = 0
    [rows, cols] = a.shape
    for i in range(0, rows):
        for j in range(0, cols):
            sum = sum + (a[i, j] - b[i, j]) ** 2

if __name__ == '__main__':
    testName = 'A'      # img filename
    noiseRatio = 0.6     # noise ratio

    # load corrupted img
    pathName = '../data/'
    corrImg = mpimg.imread(pathName + testName +'.png')

    # handle corrupted img
    if(corrImg.ndim == 2):    # grey img
        [rows, cols] = corrImg.shape
        channels = 0
    elif(corrImg.ndim == 3):  # img with RGB channel
        [rows, cols, channels] = corrImg.shape

    # noise mask
    mask = (corrImg != 0)   # correct pixel
    noiseMask = np.zeros((rows, cols, channels))
    noiseMask[mask] = 1     # true: 1, false: 0

    # standardize the corrupted img
    minX = corrImg.min()
    maxX = corrImg.max()
    corrImg = (corrImg - minX) / (maxX - minX)

    # ****** learn the coefficents in regression function ******
    basisNum = 50   # define the number of basis functions
    sigma = 0.01    # define the standard deviation
    # set the mean value of each basis function
    Phi_mu = np.linspace(start = 0, stop = cols-1, num = basisNum, endpoint = True) / (cols-1)
    # here we set the standard deviation to the same value for brevity
    Phi_sigma = sigma * np.ones((1, basisNum))

    # use pixel index as the independent variable in the regression function
    x = np.linspace(start = 0, stop = cols-1, num = cols, endpoint = True)
    x = x / (cols -1)    # standardize

    # initialize resImg
    resImg = corrImg.copy()

    # for each channel
    for k in range(0, channels):
        # for each row
        for i in range(0, rows):
            # select the missing pixels each row
            msk = noiseMask[i, : , k].copy()
            # select miss index and correct index
            misIdx = np.where(msk == 0)  # miss pixel index
            misNum = misIdx[0].shape[0]  # miss pixel number
            ddIdx = np.where(msk == 1)   # correct pixel index
            ddNum = ddIdx[0].shape[0]    # correct pixel number

            # compute the coefficients / weight
            Phi = np.hstack((np.ones((ddNum, 1)), np.zeros((ddNum, basisNum-1))))
            for j in range(1, basisNum):
                Phi[:, j] = normPdf(np.transpose(x[ddIdx[0]]), Phi_mu[j-1], Phi_sigma[0][j-1])

            w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Phi), Phi)), np.transpose(Phi)), np.transpose(corrImg[i, ddIdx[0], k]))

            # restore the missing values
            Phi1 = np.hstack((np.ones((misNum, 1)), np.zeros((misNum, basisNum - 1))))
            for j in range(1, basisNum):
                Phi1[:, j] = normPdf(np.transpose(x[misIdx[0]]), Phi_mu[j - 1], Phi_sigma[0][j - 1])

            resImg[i, misIdx[0], k] = np.dot(np.transpose(w), np.transpose(Phi1))

    for k in range(0, channels):
        for i in range(0, rows):
            for j in range(0, cols):
                if(resImg[i, j, k] < 0):
                    resImg[i, j, k] = 0
                elif(resImg[i, j, k] > 1):
                    resImg[i, j, k] = 1

    # show the corrupted img and restored img
    plt.imshow(corrImg)
    plt.show()
    plt.imshow(resImg)
    plt.show()

    # store img
    plt.imshow(resImg)
    plt.axis('off')
    plt.savefig(pathName + testName +'_restored.png')










