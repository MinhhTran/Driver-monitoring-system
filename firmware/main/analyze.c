#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    FILE *file = fopen("raw.txt", "r");
    char line[512];
    int tp = 0, fp = 0, tn = 0, fn = 0;
    double total_latency = 0.0;
    long long total_sram = 0;
    long long total_psram = 0;

    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "METRICS") != NULL) {
            char *status_ptr = strstr(line, "Status: ");
            if (status_ptr != NULL) {
                status_ptr += 8;
                if (strncmp(status_ptr, "TP", 2) == 0) {
                    tp++;
                } else if (strncmp(status_ptr, "FP", 2) == 0) {
                    fp++;
                } else if (strncmp(status_ptr, "TN", 2) == 0) {
                    tn++;
                } else if (strncmp(status_ptr, "FN", 2) == 0) {
                    fn++;
                }
            }

            char *lat_ptr = strstr(line, "Latency: ");
            if (lat_ptr != NULL) {
                float latency = 0.0;
                sscanf(lat_ptr, "Latency: %f ms", &latency); 
                total_latency += latency;
            }

            char *sram_ptr = strstr(line, "Free SRAM: ");
            if (sram_ptr != NULL) {
                long long sram = 0;
                sscanf(sram_ptr, "Free SRAM: %lld", &sram);
                total_sram += sram;
            }

            char *psram_ptr = strstr(line, "Free PSRAM: ");
            if (psram_ptr != NULL) {
                long long psram = 0;
                sscanf(psram_ptr, "Free PSRAM: %lld", &psram);
                total_psram += psram;
            }
        }
    }

    fclose(file);

    int total_frames = tp + fp + tn + fn;
    if (total_frames == 0) {
        printf("No valid METRICS lines found in the file.\n");
        return 1;
    }

    double tpr = 0.0;
    if ((tp + fn) > 0) {
        tpr = (double)tp / (tp + fn);
    }

    double fpr = 0.0;
    if ((fp + tn) > 0) {
        fpr = (double)fp / (fp + tn);
    }

    double avg_latency = total_latency / total_frames;
    double avg_sram = (double)total_sram / total_frames;
    double avg_psram = (double)total_psram / total_frames;

    printf("========== DMS METRICS ANALYSIS ==========\n");
    printf("Total Frames Evaluated: %d\n", tp + fp + tn + fn);
    printf("------------------------------------------\n");
    printf("True Positives  (TP) : %d\n", tp);
    printf("False Negatives (FN) : %d\n", fn);
    printf("True Negatives  (TN) : %d\n", tn);
    printf("False Positives (FP) : %d\n", fp);
    printf("------------------------------------------\n");
    printf("True Positive Rate (TPR) : %.2f%%\n", tpr * 100.0);
    printf("False Positive Rate (FPR): %.2f%%\n", fpr * 100.0);
    printf("Average Latency   : %.2f ms\n", avg_latency);
    printf("Average Free SRAM : %.0f Bytes (%.2f KB)\n", avg_sram, avg_sram / 1024.0);
    printf("Average Free PSRAM: %.0f Bytes (%.2f MB)\n", avg_psram, avg_psram / (1024.0 * 1024.0));
    printf("==========================================\n");

    return 0;
}