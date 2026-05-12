from os.path import join as pjoin
import pandas as pd
import ast

def extract_pfams_from_clan(clan):
    # Read which pfams are included in each CLAN
    clan_pfam = pd.read_csv('info_files/clan_pfam_data.csv')[['Clans', 'Pfam_List_in_combine']]
    clan_pfam.set_index("Clans", inplace=True, drop=True)
    pfam_list = clan_pfam.loc[clan, 'Pfam_List_in_combine'].replace('\', \'', ' ')[2:-2].split()
    return pfam_list


def extract_pfams_from_combine():
    # Read which pfams are included in combine    
    with open('info_files/combine_noclans.txt', "r") as f:
        lines = f.readlines()
    pfam_list = [line.strip() for line in lines]
    return pfam_list


def get_pfam_and_model_combination(file):
    # Read information related to the model from the file with the given structural content.
    # Mainly retrieves Pfam IDs with corresponding fine-tuned model parameter names and paths
    # Note: If a PFAM has both 0.5 and 0.9 fine-tuning, later entries would replace earlier ones.
    # Corrected by setting the key to include both pfam and cutoff information
    
    pfam_and_model = {}
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.split('\t')  # Remove newline characters from each element
            data_source = line[1]
            clus_coef = '_' + line[2]

            if data_source[:2] == 'PF':
                model_path = pjoin('tune_model' + clus_coef, data_source, data_source + clus_coef + '.pth')
                # for example 'tune_model_0.5/PF00001/PF00001_0.5.pth', if there is any difference, please correct.
                model_name = data_source + clus_coef  # PF0001_0.5
                pfam_and_model[model_name] = model_path
            elif data_source[:2] == 'CL':
                model_path = pjoin('tune_model' + clus_coef + '_clans', data_source, data_source + clus_coef + '.pth')
                # for example 'tune_model_0.5_clans/CL0145/CL0145_0.5.pth', if there is any difference, please correct.
                pfam_name_list = extract_pfams_from_clan(data_source)
                for pfam_name in pfam_name_list:
                    model_name = pfam_name + '_' + data_source + clus_coef # PF0001_CL0001_0.5
                    pfam_and_model[model_name] = model_path
            elif data_source == 'combine':
                model_path = pjoin('tune_model_combine', data_source + clus_coef, data_source + clus_coef + '.pth')
                pfam_name_list = extract_pfams_from_combine()
                for pfam_name in pfam_name_list:
                    model_name = pfam_name + '_' + data_source + clus_coef # PF0001_combine_0.5
                    pfam_and_model[model_name] = model_path

    return pfam_and_model


def gen_seq_list(pfam_human, human_protein, pfam):
    # Generate seqs based on the read in files
    ids_list = ast.literal_eval(pfam_human.loc[pfam, 'human_uniprot_id']) 

    seq_list = []
    for each in ids_list:
        seq = human_protein.loc[each, 'protein_sequence']
        seq_list.append(seq)

    return ids_list, seq_list
