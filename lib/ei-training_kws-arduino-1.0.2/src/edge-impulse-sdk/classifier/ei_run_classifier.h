/* Edge Impulse inferencing library
 * Copyright (c) 2021 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _EDGE_IMPULSE_RUN_CLASSIFIER_H_
#define _EDGE_IMPULSE_RUN_CLASSIFIER_H_

#include "model-parameters/model_metadata.h"

#include "ei_run_dsp.h"
#include "ei_classifier_types.h"
#include "ei_classifier_smooth.h"
#include "edge-impulse-sdk/porting/ei_classifier_porting.h"
#include "model-parameters/dsp_blocks.h"



#ifdef __cplusplus
namespace {
#endif // __cplusplus


/* Private functions ------------------------------------------------------- */

/**
 * @brief      Run a moving average filter over the classification result.
 *             The size of the filter determines the response of the filter.
 *             It is now set to the number of slices per window.
 * @param      maf             Pointer to maf object
 * @param[in]  classification  Classification output on current slice
 *
 * @return     Averaged classification value
 */
extern "C" float run_moving_average_filter(ei_impulse_maf *maf, float classification)
{
    maf->running_sum -= maf->maf_buffer[maf->buf_idx];
    maf->running_sum += classification;
    maf->maf_buffer[maf->buf_idx] = classification;

    if (++maf->buf_idx >= (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW >> 1)) {
        maf->buf_idx = 0;
    }

#if (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW > 1)
    return maf->running_sum / (float)(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW >> 1);
#else
    return maf->running_sum;
#endif
}


/**
 * (Marc) Get features from one second sample sound
 * @param raw_features Raw features array
 * @param raw_features_size Size of the features array
 * @param result Object to store the results in
 * @param debug Whether to show debug messages (default: false)
 */
extern "C" EI_IMPULSE_ERROR get_one_second_features(
    signal_t *signal,
    ei::matrix_t *fmatrix,
    bool debug = false)
{

    size_t out_features_index = 0;

    for (size_t ix = 0; ix < ei_dsp_blocks_size; ix++) {
        ei_model_dsp_t block = ei_dsp_blocks[ix];

        if (out_features_index + block.n_output_features > EI_CLASSIFIER_NN_INPUT_FRAME_SIZE) {
            ei_printf("ERR: Would write outside feature buffer\n");
            return EI_IMPULSE_DSP_ERROR;
        }

        ei::matrix_t fm(1, block.n_output_features, fmatrix->buffer + out_features_index);

        int ret = block.extract_fn(signal, &fm, block.config, EI_CLASSIFIER_FREQUENCY);
        if (ret != EIDSP_OK) {
            ei_printf("ERR: Failed to run DSP process (%d)\n", ret);
            return EI_IMPULSE_DSP_ERROR;
        }

        if (ei_run_impulse_check_canceled() == EI_IMPULSE_CANCELED) {
            return EI_IMPULSE_CANCELED;
        }

        out_features_index += block.n_output_features;
    }


    if (debug) {
        ei_printf("Features :");
        for (size_t ix = 0; ix < fmatrix->cols; ix++) {
            ei_printf_float(fmatrix->buffer[ix]);
            ei_printf(" ");
        }
        ei_printf("\n");
    }

    return EI_IMPULSE_OK;
}




#ifdef __cplusplus
}
#endif // __cplusplus

#endif // _EDGE_IMPULSE_RUN_CLASSIFIER_H_
