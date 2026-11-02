#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LINE_SIZE 1024
#define MAX_HITS 16

#define SI_HIGH_ENERGY_CUT 100.0
#define SI_Z 0.0

typedef struct {
    int index;
    double energy;
    double x;
    double y;
    double z;
} Interaction;

typedef struct {
    int event_id;
    double time;
    char event_type[8];
    int sequence_length;
    Interaction hits[MAX_HITS];
    int nhits;
} TraEvent;

/* detector identification */

int detector_from_z(double z) {

    if (z == 0.0)
        return 0; // Si

    if (z == -1.125)
        return 1; // CdTe

    return -1;
}

/* coordinate transform */

double coordinate_transform(double x, double instrument_size) {
    return x + instrument_size / 2.0;
}

/* pixel index */

int get_pixel_id(double x, double pixel_size) {
    return (int)floor(x / pixel_size) - 1;
}

/* matrix index */

int get_matrix_id(int x, int y, int matrix_size) {
    return y * matrix_size + x;
}

/* inside detector */

int check_inside(double x, double y, double size) {

    if (x < 0 || x > size)
        return 0;

    if (y < 0 || y > size)
        return 0;

    return 1;
}

/* print ASCII header */

void print_header() {

    printf("\n");
    printf("██████╗  █████╗ ██████╗ ███████╗██╗███╗   ██╗ ██████╗\n");
    printf("██╔══██╗██╔══██╗██╔══██╗██╔════╝██║████╗  ██║██╔════╝\n");
    printf("██████╔╝███████║██████╔╝███████╗██║██╔██╗ ██║██║  ███╗\n");
    printf("██╔═══╝ ██╔══██║██╔══██╗╚════██║██║██║╚██╗██║██║   ██║\n");
    printf("██║     ██║  ██║██║  ██║███████║██║██║ ╚████║╚██████╔╝\n");
    printf("╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝\n\n");
}

/* write t3pa row */

void write_row(
    FILE *out,
    int event,
    int matrix,
    double energy,
    int x,
    int y,
    double time,
    int chip
) {

    fprintf(out,"%d,%d,%.3f,%d,%d,%.0f,%d\n",
        event,
        matrix,
        energy,
        x,
        y,
        time,
        chip
    );
}

int main(int argc, char *argv[]) {

    char *filename = NULL;
    char *output = NULL;
    double PSF_radius = 0.0;

    /* parse CLI arguments */

    for (int i=1;i<argc;i++) {

        if (!strcmp(argv[i],"-f"))
            filename = argv[++i];

        else if (!strcmp(argv[i],"-o"))
            output = argv[++i];

        else if (!strcmp(argv[i],"-p"))
            PSF_radius = atof(argv[++i]);
    }

    if (!filename || !output) {

        printf("Usage: parse_tra -f file.tra.gz -o output_folder [-p PSF]\n");
        return 1;
    }

    print_header();

    char command[512];
    sprintf(command,"pigz -dc %s",filename);

    FILE *fp = popen(command,"r");

    if (!fp) {
        printf("Cannot open input\n");
        return 1;
    }

    char line[LINE_SIZE];

    TraEvent event;
    event.nhits = 0;

    double energy = 0;

    long total_events = 0;
    long singles = 0;
    long double_compton = 0;
    long mult_compton = 0;
    long events_in_file = 0;
    long max_events_per_file = 1000000;
    int file_index = 0;

    /* detector parameters */

    double instrument_size_si = 6.656;
    double pix_size_si = 0.025;
    int pix_matrix_si = 256;

    double instrument_size_cdte = 2.45*3;
    double pix_size_cdte = 0.025;
    int pix_matrix_cdte = 96*3;

    //char outfile[512];
    //sprintf(outfile,"%s/0.t3pa",output);

    //FILE *out = fopen(outfile,"w");

    //fprintf(out,"Event,Matrix Index,ToT (keV),X,Y,Ns,Overflow\n");


    char outfile[512];
    sprintf(outfile, "%s%d.t3pa", output, file_index);
    FILE *out = fopen(outfile, "w");
    if (!out){
        printf("Cannot open output file %s\n", outfile);
        return 1;
    }
    fprintf(out, "Event,Matrix Index,ToT (keV),X,Y,Ns,Overflow\n");


    while (fgets(line,LINE_SIZE,fp)) {

        char tag[3];
        sscanf(line,"%2s",tag);

        if (!strcmp(tag,"SE")) {
                
            total_events++;

            if (!strcmp(event.event_type,"PH") && event.nhits == 1) {

                singles++;

                Interaction h = event.hits[0];

                int detector = detector_from_z(h.z);

                double size = (detector) ? instrument_size_cdte : instrument_size_si;
                double pix  = (detector) ? pix_size_cdte       : pix_size_si;
                int matrix  = (detector) ? pix_matrix_cdte     : pix_matrix_si;

                double x = coordinate_transform(h.x,size);
                double y = coordinate_transform(h.y,size);

                if (check_inside(x,y,size)) {

                    int xp = get_pixel_id(x,pix);
                    int yp = get_pixel_id(y,pix);

                    int mid = get_matrix_id(xp,yp,matrix);

                    write_row(
                        out,
                        event.event_id,
                        mid,
                        h.energy,
                        xp,
                        yp,
                        event.time*1e9,
                        detector
                    );
                }
            }

            else if (!strcmp(event.event_type,"CO")) {

                if (event.sequence_length == 2)
                    double_compton++;

                else if (event.sequence_length > 2)
                    mult_compton++;
            }
        event.nhits = 0;
        }

        else if (!strcmp(tag,"ID")) {

            sscanf(line,"ID %d",&event.event_id);
        }

        else if (!strcmp(tag,"TI")) {

            sscanf(line,"TI %lf",&event.time);
        }

        else if (!strcmp(tag,"ET")) {

            sscanf(line,"ET %s",event.event_type);
        }

        else if (!strcmp(tag,"SQ")) {

            sscanf(line,"SQ %d",&event.sequence_length);
        }

        else if (!strcmp(tag,"PE")) {

            sscanf(line,"PE %lf",&energy);
        }

        else if (!strcmp(tag,"PP")) {

            Interaction h;

            sscanf(line,"PP %lf %lf %lf",&h.x,&h.y,&h.z);

            if (h.z==SI_Z && energy>SI_HIGH_ENERGY_CUT)
                continue;

            h.energy=energy;
            event.hits[event.nhits++]=h;
        }

        else if (!strcmp(tag,"CH")) {

            Interaction h;

            sscanf(line,"CH %d %lf %lf %lf %lf",
                &h.index,
                &h.x,
                &h.y,
                &h.z,
                &h.energy
            );

            if (h.z==SI_Z && h.energy>SI_HIGH_ENERGY_CUT)
                continue;

            event.hits[event.nhits++]=h;
        }

        /* process event */

        if (event.nhits>0 && !strcmp(tag,"SE")) {

            if (!strcmp(event.event_type,"PH") && event.nhits==1) {

                singles++;

                Interaction h=event.hits[0];

                int detector=detector_from_z(h.z);

                double size=(detector)?instrument_size_cdte:instrument_size_si;
                double pix=(detector)?pix_size_cdte:pix_size_si;
                int matrix=(detector)?pix_matrix_cdte:pix_matrix_si;

                double x=coordinate_transform(h.x,size);
                double y=coordinate_transform(h.y,size);

                if (!check_inside(x,y,size))
                    continue;

                int xp=get_pixel_id(x,pix);
                int yp=get_pixel_id(y,pix);

                int mid=get_matrix_id(xp,yp,matrix);

                write_row(
                    out,
                    event.event_id,
                    mid,
                    h.energy,
                    xp,
                    yp,
                    event.time*1e9,
                    detector
                );
                events_in_file++;
                if (events_in_file >= max_events_per_file) {
                    fclose(out);

                    file_index++;
                    events_in_file = 0;

                    sprintf(outfile,"%s/%d.t3pa",output,file_index);

                    out = fopen(outfile,"w");

                    if (!out) {
                        printf("Cannot open output file %s\n", outfile);
                        return 1;
                    }

                    fprintf(out,"Event,Matrix Index,ToT (keV),X,Y,Ns,Overflow\n");
                }
            }

            else if (!strcmp(event.event_type,"CO")) {

                if (event.sequence_length==2)
                    double_compton++;

                else if (event.sequence_length>2)
                    mult_compton++;
            }
        }
    }

    fclose(out);
    pclose(fp);

    printf("\n=============================\n");
    printf("EVENT STATISTICS\n");
    printf("=============================\n");

    printf("Total events: %ld\n",total_events);
    printf("Singles: %ld\n",singles);
    printf("Double Compton: %ld\n",double_compton);
    printf("Multi Compton: %ld\n",mult_compton);

    printf("=============================\n");

    return 0;
}
