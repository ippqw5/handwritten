#include "cnn.h"
#include <chrono>
void file_error(char* s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(0);
}

void calculator_ps(float* input)
{
    int weight_offset[3] = { 800, 51200, 4194304 };
    int beta_offset[3] = { 32, 64, 1024 };

    float* Weight_buf = (float*)calloc(4260640, sizeof(float));
    float* Beta_buf = (float*)calloc(1134, sizeof(float));

    FILE* fp_w = fopen("./parameters/weight.bin", "rb");
    if (!fp_w) file_error("weight.bin");

    FILE* fp_b = fopen("./parameters/bias.bin", "rb");
    if (!fp_b) file_error("bias.bin");

    fread(Weight_buf, sizeof(float), 4260640, fp_w);
    fread(Beta_buf, sizeof(float), 1134, fp_b);

    /*for (int i = 0; i < 1134; i++)
        printf("bias[%d]:%.17f\n",i, Beta_buf[i]);*/

    fclose(fp_w);
    fclose(fp_b);

#define MEM_LEN (16*16*32+8*8*64)
    float* Memory_buf = (float*)calloc(MEM_LEN + 1024 * 2, sizeof(float));
    float* Memory_top = Memory_buf + 1024;
    float* Memory_bottom = Memory_top + MEM_LEN;
    memcpy(Memory_top, input, 32 * 32 * 1 * sizeof(float));

    float* in_ptr[4];
    float* out_ptr[4];

    in_ptr[0] = Memory_top;
    out_ptr[0] = Memory_bottom - 16 * 16 * 32;

    in_ptr[1] = out_ptr[0];
    out_ptr[1] = Memory_top;

    in_ptr[2] = out_ptr[1];
    out_ptr[2] = Memory_bottom - 1024;

    in_ptr[3] = out_ptr[2];
    out_ptr[3] = Memory_top;

    int i;
    int woffset = 0;
    int boffset = 0;
    int TR, TC, TM, TN;
    int mLoops, nLoops;

    for (i = 0; i < 4; ++i)
    {
        if (i == 0)
        {
            // printf("Conv0\n");

            TR = 32;
            TC = 32;

            TM = 32;
            TN = 1;

            mLoops = 1;
            nLoops = 1;

            auto start = std::chrono::high_resolution_clock::now();
            detection_acc(in_ptr[i], out_ptr[i], Weight_buf + woffset, Beta_buf + boffset,
                1, 32, 5, 1, TM, TN, TR, TC, mLoops, nLoops, 0);
            auto end= std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            
            woffset += weight_offset[i];
            boffset += beta_offset[i];

            printf("Conv0 %f seconds\n", diff.count());
        }
        else if (i == 1)
        {
            // printf("Conv1\n");

            TR = 16;
            TC = 16;

            TM = 32;
            TN = 4;

            mLoops = 2;
            nLoops = 8;

            auto start = std::chrono::high_resolution_clock::now();
            detection_acc(in_ptr[i], out_ptr[i], Weight_buf + woffset, Beta_buf + boffset,
                32, 64, 5, 1, TM, TN, TR, TC, mLoops, nLoops, 0);
            auto end= std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;

            woffset += weight_offset[i];
            boffset += beta_offset[i];

            printf("Conv1 %f seconds\n", diff.count());
        }
        else if (i == 2)
        {
            // printf("FC1\n");

            float reorg_out[64][8][8];

            for (int m = 0; m < 64; m++)
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++)
                    {
                        reorg_out[m][r][c] = in_ptr[2][m * 64 + r * 8 + c];
                        //printf("reo1:%d\n", m * 64 + r * 8 + c);
                    }

            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                    for (int m = 0; m < 64; m++)
                    {
                        in_ptr[2][r * 512 + c * 64 + m] = reorg_out[m][r][c];
                        //printf("reo2:%d\n", r * 512 + c * 64 + m);
                    }
            
            TR = 1;
            TC = 1;

            TM = 32;
            TN = 4;

            mLoops = 1024;
            nLoops = 1;
            auto start= std::chrono::high_resolution_clock::now();
            detection_acc(in_ptr[i], out_ptr[i], Weight_buf + woffset, Beta_buf + boffset,
                8 * 8 * 64, 1024, 1, 1, TM, TN, TR, TC, mLoops, nLoops, 1);
            auto end= std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            woffset += weight_offset[i];
            boffset += beta_offset[i];
            printf("FC1 %f seconds\n", diff.count());

        }
        else if (i == 3)
        {
            printf("FC2\n");

            int m, n;
            for (m = 0; m < 14; m++)
            {
                for (n = 0; n < 1024; n++)
                {
                    float tmp_add_result;

                    if (n == 0)
                        tmp_add_result = Beta_buf[1120 + m];
                    else
                        tmp_add_result = out_ptr[3][m];

                    float partial_mul = in_ptr[3][n] * Weight_buf[4246304 + m * 1024 + n];

                    out_ptr[3][m] = partial_mul + tmp_add_result;
                }
                printf("%d: %.17f\n", m, out_ptr[3][m]);
            }
        }
    }

    free(Memory_buf);
    free(Weight_buf);
    free(Beta_buf);
}

int main()
{
    printf("*****Handwritten Mathematical Calculator Test Begin!*****\n");

    float* Input_buf = (float*)calloc(1024, sizeof(float));

    FILE* fp_i = fopen("./input_imgs/input1.bin", "rb");
    if (!fp_i) file_error("input.bin");

    fread(Input_buf, sizeof(float), 1024, fp_i);

    fclose(fp_i);


    auto start = std::chrono::high_resolution_clock::now();
    calculator_ps(Input_buf);
    auto end= std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    printf("Predicted in %f seconds.\n", diff.count());
}