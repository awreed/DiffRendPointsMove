import numpy as np
from scipy.signal import resample

# Upsample the pixel voxels and then upsample the scene magnitudes to match
# Then filter the scene magnitudes and keep pixel voxels that aren't filtered out

# There is a bug... Need to beamform the entire scene and then filter
import torch
from scipy.ndimage import zoom

def UpsamplePixels(**kwargs):

    with torch.no_grad():
        RP = kwargs.get('RP', None)
        BF = kwargs.get('BF', None)
        batch = kwargs.get('batch', None)

        print("Upsampling pixels...")

        # Beamform the entire scene using all pixels
        sceneMag = BF.Beamform(RP, range(0, len(batch)), soft=False, pixels=RP.pixPos3D).abs().detach().cpu().numpy()

        x_vals = np.unique(RP.pixPos3D[:, 0]).size
        y_vals = np.unique(RP.pixPos3D[:, 1]).size
        z_vals = np.unique(RP.pixPos3D[:, 2]).size

        sceneMag = sceneMag.reshape((x_vals, y_vals, z_vals))
        print(sceneMag.shape)

        # Upsample the pixel dimensions
        RP.pixDim[0] = 2 * RP.pixDim[0]
        RP.pixDim[1] = 2 * RP.pixDim[1]
        RP.pixDim[2] = 2 * RP.pixDim[2]

        print("New resolution is " + str(RP.pixDim[0]) + "X" + str(RP.pixDim[1]) + "X" + str(RP.pixDim[2]))

        xVect = np.linspace(RP.sceneDimX[0], RP.sceneDimX[1], RP.pixDim[0])
        yVect = np.linspace(RP.sceneDimY[0], RP.sceneDimY[1], RP.pixDim[1])
        zVect = np.linspace(RP.sceneDimZ[0], RP.sceneDimZ[1], RP.pixDim[2])

        (x, y, z) = np.meshgrid(xVect, yVect, zVect)

        upsampledPixels = np.hstack((np.reshape(x, (np.size(x), 1)), np.reshape(y, (np.size(y), 1)),
                                   np.reshape(z, (np.size(z), 1))))

        # Update the pixel dimensions
        RP.pixPos3D = upsampledPixels

        newDim, _ = upsampledPixels.shape

        print("here")
        sceneMagUpsampled = zoom(sceneMag, (2, 2, 2))
        sceneMagUpsampled = sceneMagUpsampled.reshape((newDim, ))

        u = np.mean(sceneMagUpsampled)
        std = np.std(sceneMagUpsampled)
        w = 0

        # Filter the scene magnitudes, find a threshold size so that pixels fit into memory
        while np.count_nonzero(~np.isnan(sceneMagUpsampled)) > 15500:
            sceneMagUpsampled[sceneMagUpsampled[:] < u + w*std] = np.nan
            w += .1

        # Keep the pixels where scene magnitude was above threshold

        indices = np.where(~np.isnan(sceneMagUpsampled))
        print("Working with this many pixels: ")
        print(np.count_nonzero(~np.isnan(sceneMagUpsampled)))
        pixels = np.squeeze(upsampledPixels[indices, :])
        print(pixels.shape)

        return pixels




