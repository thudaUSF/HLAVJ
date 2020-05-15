import traceback
import pandas as pd
import os
import sys
from lifelines.statistics import logrank_test
from lifelines.utils import median_survival_times #I couldn't get this to work
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
from lifelines import CoxPHFitter

"""
Overall, finds significant survival of people w/ certain HLA, V- or J- segments, and HLA/V or HLA/J segments. Plots data for p<.05.
Pre-requisite files - aliquot.tsv, clinical.tsv, HLA.txt, TRB.csv (or any other T-Cell receptor). 
    aliquot.tsv has many headers, downloaded from GDC portal repository
        Only headers needed are 'aliquot_id' (Filename),'case_submitter_id' (Case), 'aliquot_submitter_id' (For the MMRF dataset, this has the origin attached to it)
    clinical.tsv has many headers, downloaded from GDC portal respository
        Only headers needed are 'submitter_id' (Case), 'days_to_death' (-- if did not, number if is), 'days_to_last_known_disease_status'
    HLA.txt has no default headers in input file
        Columns are filenames, then all HLA alleles (note that each HLA has two alleles)
        After matching HLA to case, if there are duplicate typings (e.g. HLA-typing two samples (blood, tumor) from the same individual), will drop duplicates, so each person has only one HLA.
        Program only uses survival for those who are HLA-typed (for example, if some are missing, it accounts for this)
        Also drops people who did not have a MHC II HLA for example.
    TRB.csv has a few unneeded headers
        Before running this program, be sure to filter V Match Length, J Match Length, V Match Percent, J Match Percent by 19, 19, 90, 90 (or whatever preferred.)
        Headers Needed: Filename, VID, JID. 
Workflow:
1. Fix input files and create reference sheets.
2. Filter by selected origin. This is used for V-, J-segments and HLA/V- or J- combinations to match selected origin.
3. Count frequency of HLA types. True frequency. (Removes duplicates if a person has the allele twice)
4. Count frequency of V- or J-segments. True frequency.
5. Appends HLA types to V- J-segments to create combination.
6. Find frequency of combinations (not true frequency because one would have to drop specific combos that are duplicate in each case)
7. Create checklist of all things to check (HLA, VJ segment, Combination) based on selected frequencies.
8. Find survival log-rank test for HLA (Prints HLA significant cases) , VJ segment, and Combination. KM Plots for significant cases.
9. Print all significant cases. Outputs to ps.csv. Prints arms for significant combinations. Outputs to pscombos,arms.csv.

"""

class KMF(object): #Used to graph group vs. control in Kaplan-Meier Curve; Can I write KMF() without object?
    def __init__(self, group1, group2, name, savename):
        self.group1 = group1
        self.group2 = group2
        self.name = name
        self.savename = savename
        
    def KMFplot(self):    
        kmf.fit(self.group1['Days to Last Known Disease Status'], self.group1['Death'], label = self.name) # timeline=range(0,100,2) is an option that can be used to manually set x interval
        ax = kmf.plot(ci_show=True) #can be ci_show=False to remove Confidence Intervals; Exponential Greenwood Confidential Interval used, 95% CI; doesn't show if sample size is too low
        kmf.fit(self.group2['Days to Last Known Disease Status'], self.group2['Death'], label = 'control') 
        ax = kmf.plot(ax=ax, ci_show=True)    
        ax.set_ylabel('% Population')
        ax.set_xlabel('Surival (Days)')
        ax.set_ylim([0,1.1])
        #ax.set_title('Comparison of those with ' + item + 'recombination vs. HLA-typed Multiple Myeloma Population without this recombination') #Long Title doesn't format well
        ax.get_figure().savefig(self.savename + ".png") #Saves to where code is run
        ax.get_legend().remove()
        ax.remove()

path = "/Users/thuda/Desktop/Research/4-MMRF/VDJ/4-Everything/" 

#Takes aliquot thing, ends up with filename, case, Submitter id (which is used to identify Origin) 
aliquot = pd.read_csv(path + "aliquot.tsv",sep='\t')
reference = pd.DataFrame() #reference with filename, case, Origin
reference = pd.concat([aliquot['aliquot_id'],aliquot['case_submitter_id'],aliquot['aliquot_submitter_id']],axis=1) #example: b6asd800-4c3a-4158-a7af-5243b485ee78, MMRF_2007, MMRF_2007_3_BM_CD138pos_T1_KHS5U_L16538
reference.rename(columns={'aliquot_id':'Filename', 'case_submitter_id':'Case'}, inplace=True)

#allcaseOrigin has filename, case, Origin1 (e.g. PB), Origin2 (e.g. Whole); Origins are separated and defined from filename. allcaseOrigin filters all filenames that came from a specific origin.
Origin = reference['aliquot_submitter_id'].str.split("_", expand=True) #Throwaway series that just has the filename split (so that origin can be found)
reference['Origin1'] = Origin.iloc[:,3] #Define new columns based on where origins are usually in filename
reference['Origin2'] = Origin.iloc[:,4]
reference = reference.drop(columns = ['aliquot_submitter_id'])
Origin1 = "PB"
Origin2 = "Whole"
notOrigin2 = "CD138pos"
is_Origin1 = reference['Origin1']==Origin1
is_Origin2 = reference['Origin2']==Origin2
#is_Origin2 = reference['Origin2']!=notOrigin2 #if avoiding a specific one
allcaseOrigin = reference[is_Origin1&is_Origin2]

#Takes clinical data directly, corrects to numeric given usual gdc format; sets up cox for later via clinicalcox dataframe 
clinical = pd.read_csv(path + "clinical.tsv", sep='\t')
clinical.rename(columns={'submitter_id':'Case', 'age_at_index':'Age', 'gender':'Gender', 'iss_stage':'Stage','days_to_death':'Death', 'days_to_last_known_disease_status':'Days to Last Known Disease Status'}, inplace=True)
clinical = clinical[['Case','Age','Gender','Stage','Days to Last Known Disease Status', 'Death']].copy()
clinical = clinical.drop_duplicates(subset = 'Case', keep = 'first', inplace = False) #Remove duplicates from clinical file
clinical['Death'] = clinical['Death'].where(clinical['Death'] == '--',1) #if did die, write 1
clinical['Death'] = clinical['Death'].mask(clinical['Death'] == '--',0) # if didn't die, write 0
clinical['Death'] = pd.to_numeric(clinical['Death']) #make sure it's numeric
clinicalcox = clinical.reset_index(drop = True) 
clinical = clinical[['Case','Days to Last Known Disease Status', 'Death']].copy()

#General Cox for gender, age, stage; drop unknown values
stage_dict = {'Unknown': '-','I': 1,'II': 2,'III': 3} 
clinicalcox = clinicalcox.replace({'Stage': stage_dict})
clinicalcox = clinicalcox.replace("male", 1)
clinicalcox = clinicalcox.replace("female", 0)
Caseclinicalcox = clinicalcox['Case']
clinicalcox = clinicalcox.apply(pd.to_numeric, errors='coerce')
clinicalcox['Case'] = Caseclinicalcox
clinicalcox = clinicalcox.dropna()
Usedclinicalcox = clinicalcox.drop(columns = ['Case'])
"""
cox = CoxPHFitter()
cox.fit(Usedclinicalcox,duration_col='Days to Last Known Disease Status',event_col='Death')
cox.print_summary() #exp(coef) column indicates that the event/risk factor such as gender can increase/decrease survival
"""

#Fixes HLA format
HLA = pd.read_csv(path + "results.txt",names = ['Filename', 'HLA-A', 'HLA-A\'', 'HLA-B', 'HLA-B\'', 'HLA-C', 'HLA-C\'', 'HLA-DPB1', 'HLA-DPB1\'', 'HLA-DQB1', 'HLA-DQB1\'', 'HLA-DRB1', 'HLA-DRB1\''], sep='\t')
HLA['Filename'] = HLA['Filename'].str.replace('_wxs_gdc_realn.bam', '') 
HLA.drop_duplicates(subset = 'Filename', keep = 'first', inplace = False) #remove HLA duplicates

#Give each Case an HLA first. Some Cases might end up with two HLAs, so drop duplicates. Then make sure for each filename under a Case, each one has the same HLA. 
HLA = HLA.merge(reference[['Filename','Case']],on='Filename', how = 'left')
allcaseOriginHLA = allcaseOrigin.merge(HLA[['Case','HLA-A', 'HLA-A\'', 'HLA-B', 'HLA-B\'', 'HLA-C', 'HLA-C\'', 'HLA-DPB1', 'HLA-DPB1\'', 'HLA-DQB1', 'HLA-DQB1\'', 'HLA-DRB1', 'HLA-DRB1\'']],on = 'Case', how = 'left') #Noted all columns to avoid Origin1, Origin2
allcaseOriginHLAdrop = allcaseOriginHLA.drop_duplicates(subset = "Case",keep = 'first', inplace = False) #Reference file is all cases who are HLA-typed with the specificed origin
HLAdrop = HLA.drop_duplicates(subset = 'Case', keep = 'first', inplace = False) #Remove duplicates, keep first instance (in MMRF set, HLA comes from both blood/PB - these are nearly always same, so we remove duplicates)
HLAdrop = HLAdrop.drop(columns = ['Filename']) #To avoid reference + HLAdrop having two filename columns
referenceHLA = reference.merge(HLAdrop,on='Case', how = 'right') #Reference file because each person has multiple files. Make sure that each filename has a specific HLA. (Filename --> Case --> HLA)

#Takes TRB, cleans Filename,VID,JID
TRB = pd.read_csv(path + "TRBfm.csv")
TRBVJ = TRB[['Filename','VID','JID','CDR3']].copy()
TRBVJ['Filename'] = TRBVJ['Filename'].str.split("_", expand=False).str[1] #e.g. sliced_eb72c238-3349-4b49-9483-503ac9c49144_wxs_gdc_realn.bam.tsv
TRBVJ['VID'] = TRBVJ['VID'].str.split("|", expand=False).str[1] #L05149|TRBV20/OR9-2*01|Homo --> TRBV20/OR9-2*01
TRBVJ['JID'] = TRBVJ['JID'].str.split("|", expand=False).str[1] #M14159|TRBJ2-7*01|Homo --> TRBJ2-7*01

#
referencedrop = reference.drop_duplicates(subset = 'Filename', keep = 'first', inplace = False) #Just in case there are multiple filenames? 
TRBVJ = pd.merge(TRBVJ,referencedrop,on='Filename',how='left') #now TRBVJ has Case, Origins

#Filer VJ receptors by origin
is_Origin1 = TRBVJ['Origin1']==Origin1
is_Origin2 = TRBVJ['Origin2']==Origin2
#is_Origin2 = TRBVJ['Origin2']!=notOrigin2 #if avoiding
TRBVJ = TRBVJ[is_Origin1&is_Origin2]

#TRBHLAVJ will have VIDs, JIDs, and HLAs attached for the people who had a productive VID. Remove the HLA-typed people who didn't have a VJ combination.
TRBHLAVJ = pd.merge(TRBVJ,HLAdrop, on = 'Case', how = 'left')
TRBHLAVJ = TRBHLAVJ[TRBHLAVJ['VID'].notna()]
#print(TRBHLAVJ.head())
print("Preparation 1 Done")

#Viral CDR3
Viralcdr3 = pd.read_csv("/Users/thuda/Desktop/Research/4-MMRF/VDJ/8-ViralCDR3/ViralCDR3TRB.csv")
Viruses = Viralcdr3['Epitope species'].drop_duplicates(keep = 'first', inplace = False)
Viruses = Viruses.reset_index(drop=True)
Usedv = pd.DataFrame()
Used = pd.DataFrame()
psviral = pd.DataFrame()
allpsviral = pd.DataFrame()
Cases = pd.DataFrame()
RCases = pd.DataFrame() 
Casesdrop = pd.DataFrame()
p = pd.DataFrame()
coxsummary = pd.DataFrame()
OS_cox_p_age = []
OS_cox_bcoef_age = []
OS_cox_p_gender = []
OS_cox_bcoef_gender = []
OS_cox_p_stage = []
OS_cox_bcoef_stage = []
Virusescox = []
RreferenceHLAdropOrigin = allcaseOriginHLAdrop.sample(frac = .5)
#print(RreferenceHLAdropOrigin.shape)
#RreferenceHLAdropOrigin.to_csv(path + "RreferenceHLAdropOrigin.csv")
RTRBHLAVJ = TRBHLAVJ[TRBHLAVJ['Case'].isin(RreferenceHLAdropOrigin['Case'])] 
for index, item in Viruses.items():
    is_virus = Viralcdr3['Epitope species']==item
    Usedv = Viralcdr3[is_virus]
    Used = TRBHLAVJ[['Case','CDR3']].copy()
    RUsed = RTRBHLAVJ[['Case','CDR3']].copy()
    Cases = Used[Used['CDR3'].isin(Usedv['CDR3'])] # is item found in Column 1 or 2 (TRB-J/HLA-A or TRB-J/HLA-A')
    RCases = RUsed[RTRBHLAVJ['CDR3'].isin(Usedv['CDR3'])]
    Casesdrop = Cases.drop_duplicates(subset = "Case", keep = 'first', inplace = False) #Remove duplicates, keep first instance (doesn't matter because instances are practically same. (We only look at Case ID anyway)
    RCasesdrop = RCases.drop_duplicates(subset = "Case",keep = 'first',inplace = False)
    save = item.translate(str.maketrans('', '', '*:\'\/')) #Problem with saving file if there's asterisk/semicolon
    CompCasesdrop = allcaseOriginHLAdrop[~allcaseOriginHLAdrop['Case'].isin(Casesdrop['Case'])] #Remove rows where the case is found in the Observed group
    RCompCasesdrop = RreferenceHLAdropOrigin[~RreferenceHLAdropOrigin['Case'].isin(RCasesdrop['Case'])]
    Casesdrop = Casesdrop.reset_index(drop=True)
    RCasesdrop = RCasesdrop.reset_index(drop=True)
    CompCasesdrop = CompCasesdrop.reset_index(drop=True)  
    Casesdropsurv = Casesdrop.merge(clinical,on='Case', how='left')
    CompCasesdropsurv = CompCasesdrop.merge(clinical,on='Case',how='left')
    RCompCasesdrop = RCompCasesdrop.reset_index(drop=True)  
    RCasesdropsurv = RCasesdrop.merge(clinical,on='Case', how='left')
    RCompCasesdropsurv = RCompCasesdrop.merge(clinical,on='Case',how='left')

    #Casesdropsurv.to_csv(path + save + "-casessurv.csv", index=False)
    #CompCasesdropsurv.to_csv(path + save + "-compcasessurv.csv", index=False) #Save Complement of Cases with no duplicates and no empty HLA

    results = logrank_test(Casesdropsurv['Days to Last Known Disease Status'], CompCasesdropsurv['Days to Last Known Disease Status'], Casesdropsurv['Death'], CompCasesdropsurv['Death'])
    #results.print_summary() #details of each
    datap = [[item, results.p_value, Casesdropsurv.shape[0], CompCasesdropsurv.shape[0]]]
    p = pd.DataFrame(datap, columns=['Combination','Log-rank Test p-value', 'Sample Size', 'Complement Sample Size'])
    allpsviral = allpsviral.append(p,ignore_index=True)
    
    if results.p_value < .05:
        Rresults = logrank_test(RCasesdropsurv['Days to Last Known Disease Status'], RCompCasesdropsurv['Days to Last Known Disease Status'], RCasesdropsurv['Death'], RCompCasesdropsurv['Death'])
        Rdatap = [[item, Rresults.p_value, RCasesdropsurv.shape[0], RCompCasesdropsurv.shape[0]]]
        Rp = pd.DataFrame(Rdatap, columns=['Combination','Log-rank Test p-value', 'Sample Size', 'Complement Sample Size'])
        psviral = psviral.append(p,ignore_index=True)
        psviral = psviral.append(Rp, ignore_index=True)
        #Casesdropsurv.to_csv(path + save + "-casessurv.csv", index=False)
        #CompCasesdropsurv.to_csv(path + save + "-compcasessurv.csv", index=False)
        KMFit = KMF(Casesdropsurv, CompCasesdropsurv, item, save)
        KMFit.KMFplot()
    """
    if results.p_value < .05 and len(Casesdrop.index) > 20: #20 is an arbitrary limit
        Casesdropsurvcox = Casesdrop.merge(clinicalcox,on='Case', how='left')
        CompCasesdropsurvcox = CompCasesdrop.merge(clinicalcox,on='Case',how='left')
        Casesdrop = Casesdropsurvcox[['Age','Gender','Stage','Days to Last Known Disease Status', 'Death']].copy()
        Casesdrop = Casesdrop.dropna()
        cox = CoxPHFitter()
        try:
            cox.fit(Casesdrop,duration_col='Days to Last Known Disease Status',event_col='Death')
            cox.print_summary(3,"ascii")
            OS_cox = cox.summary
            #print(OS_cox.head())
            #OS_cox.to_csv(path + "OS.csv")
            Virusescox.append(str(item))
            OS_cox_p_age.append(OS_cox.iloc[0][8])
            OS_cox_bcoef_age.append(OS_cox.iloc[0][1])
            OS_cox_p_gender.append(OS_cox.iloc[1][8])
            OS_cox_bcoef_gender.append(OS_cox.iloc[1][1])
            OS_cox_p_stage.append(OS_cox.iloc[2][8])
            OS_cox_bcoef_stage.append(OS_cox.iloc[2][1])
        except:
            print(traceback.format_exc())"""
            
print(allpsviral[:])
#allpsviral.to_csv(path + "psviral.csv")
print(psviral[:])
#psviral.to_csv(path + "onlypsviral.csv")
coxsummary["Virus"]=Virusescox
coxsummary["OS_cox_p_age"] = OS_cox_p_age
coxsummary["OS_cox_bcoef_age"] = OS_cox_bcoef_age
coxsummary["OS_cox_p_gender"] = OS_cox_p_gender
coxsummary["OS_cox_bcoef_gender"] = OS_cox_bcoef_gender
coxsummary["OS_cox_p_stage"] = OS_cox_p_stage
coxsummary["OS_cox_bcoef_stage"] = OS_cox_bcoef_stage
#coxsummary.to_csv(path + "Viralcoxsummary.csv")

#Count amount (frequency) of people with a certain HLA subtype. Get total count for all HLAs in both alleles - add total counts, subtract repetitions (people who had the same allele twice.)
#Count of HLA may be slightly different than a manual calculation because people can be typed differently depending on which duplicate removed (typing difference is usually between subtypes e.g. 3:01 --> 3:05)
frequentHLAtype = pd.DataFrame()
frequentHLAtypepairs = pd.DataFrame()
frequentHLA = pd.DataFrame()
frequentHLAtotal = pd.DataFrame(columns = ['HLA','frequency'])
for columnname, columndata in HLAdrop.iteritems():
    if "HLA-" in columnname and not "\'" in columnname: #For every HLA type, don't do HLA-A'
        frequentHLAtypeSeries = HLAdrop[columnname].value_counts() #Counts frequency of values in a list
        frequentHLAtype = pd.DataFrame({columnname:frequentHLAtypeSeries.index, 'count':frequentHLAtypeSeries.values}) #Convert to Dataframe; first column = HLA type, second column = frequency
        frequentHLAtypeSeries2 = HLAdrop[columnname + "\'"].value_counts()
        frequentHLAtype2 = pd.DataFrame({columnname:frequentHLAtypeSeries2.index, 'count\'':frequentHLAtypeSeries2.values}) #Do above for second allele
        HLAtype = columnname.replace("HLA-","")
        #print(HLAtype)
        namepairs = HLAtype + 'pairs'
        HLAdrop[namepairs] = HLAdrop[columnname] + HLAdrop[columnname + "\'"] #Combine the alleles together
        frequentHLAtypepairsSeries = HLAdrop[namepairs].value_counts() #Frequency of combined alleles
        frequentHLAtypepairs = pd.DataFrame({namepairs:frequentHLAtypepairsSeries.index, 'count pairs':frequentHLAtypepairsSeries.values}) #Convert to dataframe
        frequentHLAtypepairs[columnname] = frequentHLAtypepairs[namepairs].str.split(HLAtype, expand=False).str[1]
        frequentHLAtypepairs['2'] = frequentHLAtypepairs[namepairs].str.split(HLAtype, expand=False).str[2]
        #print(frequentHLAtypepairs.head())
        is_repeat = frequentHLAtypepairs[columnname] == frequentHLAtypepairs['2'] 
        frequentHLAtypepairs = frequentHLAtypepairs[is_repeat]
        frequentHLAtypepairs = frequentHLAtypepairs.reset_index(drop = True) #Take pairs from the total count where both alleles are the same only
        frequentHLAtypepairs[columnname] = HLAtype + frequentHLAtypepairs[columnname].astype(str) #Fixing string to make it match 
        frequentHLAtype = frequentHLAtype.merge(frequentHLAtype2, on = columnname,how='left') #Merge allele2
        frequentHLAtype = frequentHLAtype.merge(frequentHLAtypepairs[[columnname,'count pairs']], on = columnname,how='left') #Merge pairing numbers
        truefreqname = 'truefrequency HLA-' + HLAtype
        frequentHLAtype[truefreqname] = frequentHLAtype.fillna(0)['count'] + frequentHLAtype.fillna(0)['count\''] - frequentHLAtype.fillna(0)['count pairs']  #add the amounts between number of alleles, subtract pairs
        frequentHLAtype = frequentHLAtype.rename(columns={columnname:'HLA', truefreqname:'frequency'}) #match it so it can be added to a list of all HLA    
        frequentHLAtype = frequentHLAtype.sort_values(by='frequency',ascending=False)
        frequentHLAtype = frequentHLAtype.reset_index(drop = True)
        #print(frequentHLAtype.head())
        frequentHLAtotal = frequentHLAtotal.append(frequentHLAtype[['HLA','frequency']], ignore_index=True) #List of all HLAs in a list and real counts
    else:
        continue
#print(frequentHLAtotal.head())
#print(frequentHLAtotal.shape)

frequentVSeries = TRBHLAVJ['VID'].value_counts() 
frequentVtotal = pd.DataFrame({'VID':frequentVSeries.index, 'count':frequentVSeries.values})
frequentJSeries = TRBHLAVJ['JID'].value_counts()
frequentJtotal = pd.DataFrame({'JID':frequentJSeries.index, 'count':frequentJSeries.values})
#print(frequentVtotal.head())
#print(frequentJtotal.head())
frequentVJtotal = pd.DataFrame(columns = ['VJID','frequency'])
frequentVtotal = frequentVtotal.rename(columns={'VID':'VJID', 'count':'frequency'})   
frequentJtotal = frequentJtotal.rename(columns={'JID':'VJID', 'count':'frequency'})    
frequentVJtotal = frequentVtotal.append(frequentJtotal[['VJID','frequency']], ignore_index=True) #List of all VJs and approximate counts

#Create HLA-V or HLA-J combination in strings in TRBHLAVJ
for columnname, columndata in TRBHLAVJ.iteritems():
    if "HLA-" in columnname:
        HLAtype = columnname.replace("HLA-","")
        TRBHLAVJ['HLA-' + HLAtype + ' + V-type'] = TRBHLAVJ[columnname] + "-" + TRBHLAVJ['VID']
        TRBHLAVJ['HLA-' + HLAtype + ' + J-type'] = TRBHLAVJ[columnname] + "-" + TRBHLAVJ['JID']
    else:
        continue
#print(TRBHLAVJ.head())

#Count frequency of HLA-V or HLA-J combinations using both alleles; same logic as HLA frequency
frequentHLAVJtotal = pd.DataFrame(columns = ['combo','frequency'])
for columnname, columndata in TRBHLAVJ.iteritems():
    if "-type" in columnname and not "\'" in columnname:
        #print(columnname)
        HLAtype = columnname.replace("HLA-","")
        fixedname = columnname
        if "V-type" in columnname: 
            HLAtype = HLAtype.replace(" + V-type","")
            comboname = "HLA-" + HLAtype + '\' + V-type'
        elif "J-type" in columnname: 
            HLAtype = HLAtype.replace(" + J-type","")
            comboname = "HLA-" + HLAtype + '\' + J-type'
        frequentHLAVJSeries = TRBHLAVJ[columnname].value_counts() 
        frequentHLAVJ = pd.DataFrame({'combo':frequentHLAVJSeries.index, 'count':frequentHLAVJSeries.values})
        frequentHLAVJSeries2 = TRBHLAVJ[comboname].value_counts()
        frequentHLAVJ2 = pd.DataFrame({'combo':frequentHLAVJSeries2.index, 'count\'':frequentHLAVJSeries2.values})
        frequentHLAVJ = frequentHLAVJ.merge(frequentHLAVJ2, on ='combo',how='left')
        relfreqname = 'relativefrequency' + columnname
        frequentHLAVJ[relfreqname] = frequentHLAVJ.fillna(0)['count'] + frequentHLAVJ.fillna(0)['count\'']  #add the amounts between number of alleles, subtract pairs
        #print(frequentHLAVJ.head())
        frequentHLAVJ = frequentHLAVJ.rename(columns={columnname:'combo', relfreqname:'frequency'})    
        #print(frequentHLAVJtotal.head())
        frequentHLAVJ = frequentHLAVJ.sort_values(by='frequency',ascending=False)
        frequentHLAVJ = frequentHLAVJ.reset_index(drop = True)
        frequentHLAVJtotal = frequentHLAVJtotal.append(frequentHLAVJ[['combo','frequency']], ignore_index=True)
    else:
        continue
#print(frequentHLAVJtotal.head())
print("Frequency + HLA-VJ attachment done")

#Make a cutoff for doing log-rank tests - note that HLA's is an exact cutoff for number of patients, VJ and HLA-VJ are inexact cutoffs because counting properly would be problematic
is_boundary_HLA = frequentHLAtotal['frequency']>=40
is_boundary_VJ = frequentVJtotal['frequency']>=40
is_boundary_HLAVJ = frequentHLAVJtotal['frequency']>=40
ChecklistHLA = pd.DataFrame(frequentHLAtotal[is_boundary_HLA])
ChecklistHLA = ChecklistHLA.reset_index(drop = True)
ChecklistVJ = pd.DataFrame(frequentVJtotal[is_boundary_VJ])
ChecklistVJ = ChecklistVJ.reset_index(drop = True)
ChecklistHLAVJ = pd.DataFrame(frequentHLAVJtotal[is_boundary_HLAVJ])    
ChecklistHLAVJ = ChecklistHLAVJ.reset_index(drop = True)
#ChecklistHLA.to_csv(path + "ChecklistHLA.csv")
#ChecklistVJ.to_csv(path + "ChecklistVJ.csv")
#ChecklistHLAVJ.to_csv(path + "ChecklistHLAVJ.csv")

Cases = pd.DataFrame() #All Cases that fit the requirements
RCases = pd.DataFrame() #Replicative set of Cases, Randomly generates half from cases
Casesdrop = pd.DataFrame() #For Cases that fit requirements, remove duplicate people (for example: multiple reads of a HLA-TRB-V on same person that match) dropped
p = pd.DataFrame() #dataframe with the p value in that run; note p < .05 for these tests
allpsHLA = pd.DataFrame() #dataframe with all p values in the HLA run (used to identify p-value of one arm from combination)
allpsVJ = pd.DataFrame() #dataframe with all p values in the VJ run (used to identify p-value of one arm from combination)
psHLA = pd.DataFrame() #dataframe with significant p values in the HLA run (used for COX)
psVJ = pd.DataFrame() #dataframe with significant p values in the VJ run (used for COX)
psHLAVJ = pd.DataFrame() #dataframe with all significant p values in the combo run
ps = pd.DataFrame() #all p-value < .05 in HLA, TRBV/J, HLA-TRBV/J

#Testing for survival differences for all HLA in Checklist HLA
RHLAdrop = HLAdrop.sample(frac = .5) #Creates half replicative set that will be used for all run throughs of HLA (All R before variable name - Replicative Set)
for index, row in ChecklistHLA.iterrows():
    item = row[0] #item = HLA being checked from checklist e.g. A*02:01
    HLAtype = item.split('*')[0]
    HLAtype = "HLA-" + HLAtype #HLA-DPB
    column1 = HLAtype
    column2 = HLAtype + '\'' 
    Used = HLAdrop[['Case',column1,column2]].copy()
    RUsed = RHLAdrop[['Case',column1,column2]].copy()
    is_c1 = Used[column1]==item
    is_c2 = Used[column2]==item
    is_Rc1 = RUsed[column1]==item
    is_Rc2 = RUsed[column2]==item
    Cases = Used[is_c1|is_c2] # is item found in Column 1 or 2 (HLA-A or HLA-A')
    RCases = RUsed[is_Rc1|is_Rc2]
    Casesdrop = Cases.drop_duplicates(subset = "Case", keep = 'first', inplace = False) #Remove duplicates (in case, probably unnecessary)
    RCasesdrop = RCases.drop_duplicates(subset = "Case", keep = 'first', inplace = False)
    #print(Casesdrop.shape)
    #print(Casesdrop.head())
    save = item.translate(str.maketrans('', '', '*:\'\/')) #Problem with saving file if there's asterisk/semicolon
    #Casesdrop.to_csv(path + save + ".csv", index=False) #Save Cases with no duplicates
    CompCases = Used[~Used['Case'].isin(Casesdrop['Case'])] #Remove rows where the case is found in the Observed group, all variables with Comp --> Complementary Set
    RCompCases = RUsed[~RUsed['Case'].isin(RCasesdrop['Case'])]
    CompCases = CompCases[CompCases[HLAtype].notnull()]
    RCompCases = RCompCases[RCompCases[HLAtype].notnull()] #Remove rows where there is no HLA to compare (for example, if HLA-A is typed but not MHC II receptors)
    #print(Casesdrop.shape)
    #CompCases.to_csv(path + save + "-beforedrop.csv", index=False) #Save Complement of Cases with no duplicates and no empty HLA
    CompCasesdrop = CompCases.drop_duplicates(subset = "Case", keep = 'first', inplace = False)
    RCompCasesdrop = RCompCases.drop_duplicates(subset = "Case", keep = 'first', inplace = False)
    #print(CompCasesdrop.shape)
    #print(Casesdrop.head())
    Casesdrop = Casesdrop.reset_index(drop=True)
    RCasesdrop = RCasesdrop.reset_index(drop=True)
    CompCasesdrop = CompCasesdrop.reset_index(drop=True)
    RCompCasesdrop = RCompCasesdrop.reset_index(drop=True)
    #CompCasesdrop.to_csv(path + save + "-Comp.csv", index=False) #Save Complement of Cases with no duplicates and no empty HLA
    
    Casesdropsurv = Casesdrop.merge(clinical,on='Case', how='left')
    RCasesdropsurv = RCasesdrop.merge(clinical,on='Case', how='left')
    #RCases.to_csv(path + save + "-Replicative.csv", index=False)
    CompCasesdropsurv = CompCasesdrop.merge(clinical,on='Case', how='left')
    RCompCasesdropsurv = RCompCasesdrop.merge(clinical,on='Case', how='left')
    #Casesdropsurv.to_csv(path + save + "-surv.csv", index=False)
    #CompCasesdropsurv.to_csv(path + save + "-Compsurv.csv", index=False) 
    results = logrank_test(Casesdropsurv['Days to Last Known Disease Status'], CompCasesdropsurv['Days to Last Known Disease Status'], Casesdropsurv['Death'], CompCasesdropsurv['Death'])
    Rresults = logrank_test(RCasesdropsurv['Days to Last Known Disease Status'], RCompCasesdropsurv['Days to Last Known Disease Status'], RCasesdropsurv['Death'], RCompCasesdropsurv['Death'])
    #print(item, "  ", results.p_value) #if you want all p-values
    #results.print_summary() #details of each
    #print("P-value of " + item) #if you want all p-values
    #print(results.p_value) #if you want all p-values    
    datap = [[item, results.p_value, Casesdrop.shape[0], CompCasesdrop.shape[0]]]
    p = pd.DataFrame(datap, columns=['Combination','Log-rank Test p-value', 'Sample Size', 'Complement Sample Size'])
    allpsHLA = allpsHLA.append(p,ignore_index=True)
    if results.p_value < .05:
        #Casesdropsurv.to_csv(path + save + "-surv.csv", index=False)
        #CompCasesdropsurv.to_csv(path + save + "-Compsurv.csv", index=False) 
        #RCasesdropsurv.to_csv(path + save + "-Replicative.csv", index=False)
        #RCompCasesdropsurv.to_csv(path + save + "-Replicativecomp.csv", index=False)
        #results.print_summary()
        Rdatap = [[item + "-Rep", Rresults.p_value, RCasesdrop.shape[0], RCompCasesdrop.shape[0]]]
        Rp = pd.DataFrame(Rdatap, columns=['Combination','Log-rank Test p-value', 'Sample Size', 'Complement Sample Size'])
        ps = ps.append(p, ignore_index=True)
        psHLA = psHLA.append(p,ignore_index=True)
        ps = ps.append(Rp, ignore_index =True)
        #KMFit = KMF(Casesdropsurv, CompCasesdropsurv, item, save)
        #KMFit.KMFplot()
        #RKMFit = KMF(RCasesdropsurv, RCompCasesdropsurv, item + "-Replicative", save + "-Replicative")
        #RKMFit.KMFplot()
#print(ps[:]) #prints results of significant values so far (just HLA)
print("HLA done")

#Replicative set for VJ, HLA-VJ combinations; also dropping duplicate for the reference set (allcaseOriginHLAdrop), which has all HLA-typed cases of the specific origin
RreferenceHLAdropOrigin = allcaseOriginHLAdrop.sample(frac = .5)
#print(RreferenceHLAdropOrigin.shape)
#RreferenceHLAdropOrigin.to_csv(path + "RreferenceHLAdropOrigin.csv")
RTRBHLAVJ = TRBHLAVJ[TRBHLAVJ['Case'].isin(RreferenceHLAdropOrigin['Case'])] #Replicative set leaves only cases which have recombined (TRBHLAVJ) and are found in the total halved set (RreferenceHLAdropOrigin)
#RTRBHLAVJ.to_csv(path + "RTRBHLAVJ.csv")

#Testing for survival differences for all VJ in ChecklistVJ; follows same logic as HLA, see there for more details
for index, row in ChecklistVJ.iterrows():
    item = row[0] #item = VJ e.g. TRBV5-1*01
    if "V" in item: 
        column = 'VID'
    elif "J" in item:
        column = 'JID' 
    Used = TRBHLAVJ[['Filename','Case',column]].copy()
    RUsed = RTRBHLAVJ[['Filename','Case',column]].copy()
    is_c = Used[column]==item
    is_Rc = RUsed[column]==item
    Cases = Used[is_c] # is item found in Column 1 or 2 (TRB-J/HLA-A or TRB-J/HLA-A')
    RCases = RUsed[is_Rc] 
    Casesdrop = Cases.drop_duplicates(subset = "Case", keep = 'first', inplace = False) #Remove duplicates, keep first instance (doesn't matter because instances are practically same. (We only look at Case ID anyway)
    RCasesdrop = RCases.drop_duplicates(subset = "Case", keep = 'first', inplace = False)
    save = item.translate(str.maketrans('', '', '*:\'\/')) #Problem with saving file if there's asterisk/semicolon
    #Casesdrop.to_csv(path + save + ".csv", index=False) #Save Cases with no duplicates
    CompCases = allcaseOriginHLA[~allcaseOriginHLA['Case'].isin(Casesdrop['Case'])] #Remove rows where the case is found in the Observed group
    RCompCases = RreferenceHLAdropOrigin[~RreferenceHLAdropOrigin['Case'].isin(RCasesdrop['Case'])]
    CompCases = CompCases[CompCases['HLA-A'].notnull()] #Remove rows where there is no HLA to compare
    RCompCases = RCompCases[RCompCases['HLA-A'].notnull()]
    #print(Casesdrop.shape)
    #CompCases.to_csv(path + save + "-beforedrop.csv", index=False) #Save Complement of Cases with no duplicates and no empty HLA
    CompCasesdrop = CompCases.drop_duplicates(subset = "Case", keep = 'first', inplace = False)
    RCompCasesdrop = RCompCases.drop_duplicates(subset = "Case", keep = 'first', inplace = False)
    #print(CompCasesdrop.shape)
    #print(Casesdrop.head())
    Casesdrop = Casesdrop.reset_index(drop=True)
    CompCasesdrop = CompCasesdrop.reset_index(drop=True)
    RCasesdrop = RCasesdrop.reset_index(drop=True)
    RCompCasesdrop = RCompCasesdrop.reset_index(drop=True)
    #CompCasesdrop.to_csv(path + save + "-Comp.csv", index=False) #Save Complement of Cases with no duplicates and no empty HLA

    Casesdropsurv = Casesdrop.merge(clinical,on='Case', how='left')
    RCasesdropsurv = RCasesdrop.merge(clinical,on='Case', how='left')
    
    #RCases.to_csv(path + save + "-Replicative.csv", index=False)
    CompCasesdropsurv = CompCasesdrop.merge(clinical,on='Case', how='left')
    RCompCasesdropsurv = RCompCasesdrop.merge(clinical,on='Case', how='left')
    #Casesdropsurv.to_csv(path + save + "-surv.csv", index=False)
    #CompCasesdropsurv.to_csv(path + save + "-Compsurv.csv", index=False) 
    results = logrank_test(Casesdropsurv['Days to Last Known Disease Status'], CompCasesdropsurv['Days to Last Known Disease Status'], Casesdropsurv['Death'], CompCasesdropsurv['Death'])
    datap = [[item, results.p_value, Casesdrop.shape[0], CompCasesdrop.shape[0]]]
    p = pd.DataFrame(datap, columns=['Combination','Log-rank Test p-value', 'Sample Size', 'Complement Sample Size'])
    allpsVJ = allpsVJ.append(p,ignore_index=True)
    #print(item, "  ", results.p_value)
    #results.print_summary() #details of each
    #print("P-value of " + item) #if you want all p-values
    #print(results.p_value) #if you want all p-values    
    if results.p_value < .05:
        #Casesdropsurv.to_csv(path + save + "-surv.csv", index=False)
        #CompCasesdropsurv.to_csv(path + save + "-Compsurv.csv", index=False) 
        #results.print_summary()
        Rresults = logrank_test(RCasesdropsurv['Days to Last Known Disease Status'], RCompCasesdropsurv['Days to Last Known Disease Status'], RCasesdropsurv['Death'], RCompCasesdropsurv['Death'])  
        Rdatap = [[item + "-Rep", Rresults.p_value, RCasesdropsurv.shape[0], RCompCasesdropsurv.shape[0]]]
        Rp = pd.DataFrame(Rdatap, columns=['Combination','Log-rank Test p-value', 'Sample Size', 'Complement Sample Size'])
        ps = ps.append(p, ignore_index=True)
        psVJ = psVJ.append(p,ignore_index=True)
        ps = ps.append(Rp, ignore_index =True)
        #KMFit = KMF(Casesdropsurv, CompCasesdropsurv, item, save)
        #KMFit.KMFplot()
        #RKMFit = KMF(RCases, CompCasesdropsurv, item + "-Replicative", save + "-Replicative")
        #RKMFit.KMFplot()
print("VJ Done")

#Testing for survival differences for all HLA-VJ in ChecklistHLAVJ; follows same logic as HLA, see there for more details
for index, row in ChecklistHLAVJ.iterrows():
    item = row[0] #item = TRB-V/HLA Combination e.g. A*02:01-TRBV5-1*01
    HLAtype = item.split('*')[0]
    HLAtype = "HLA-" + HLAtype #HLA-DPB
    if "V" in item: 
        column1 = HLAtype + ' + V-type'
        column2 = HLAtype + '\' + V-type'
    elif "J" in item:
        column1 = HLAtype + ' + J-type'
        column2 = HLAtype + '\' + J-type'
    Used = TRBHLAVJ[['Filename','Case',column1,column2]].copy()
    RUsed = RTRBHLAVJ[['Filename','Case',column1,column2]].copy()
    is_c1 = Used[column1]==item
    is_c2 = Used[column2]==item
    is_Rc1 = RUsed[column1]==item
    is_Rc2 = RUsed[column2]==item
    Cases = Used[is_c1|is_c2] # is item found in Column 1 or 2 (TRB-J/HLA-A or TRB-J/HLA-A')
    RCases = RUsed[is_Rc1|is_Rc2]
    Casesdrop = Cases.drop_duplicates(subset = "Case", keep = 'first', inplace = False) #Remove duplicates, keep first instance (doesn't matter because instances are practically same. (We only look at Case ID anyway)
    RCasesdrop = RCases.drop_duplicates(subset = "Case", keep = 'first', inplace = False)
    save = item.translate(str.maketrans('', '', '*:\'\/')) #Problem with saving file if there's asterisk/semicolon
    #Casesdrop.to_csv(path + save + ".csv", index=False) #Save Cases with no duplicates
    CompCases = allcaseOriginHLA[~allcaseOriginHLA['Case'].isin(Casesdrop['Case'])] #Remove rows where the case is found in the Observed group
    RCompCases = RreferenceHLAdropOrigin[~RreferenceHLAdropOrigin['Case'].isin(RCasesdrop['Case'])]
    CompCases = CompCases[CompCases[HLAtype].notnull()] #Remove rows where there is no HLA to compare
    RCompCases = RCompCases[RCompCases[HLAtype].notnull()]
    #print(Casesdrop.shape)
    #CompCases.to_csv(path + save + "-beforedrop.csv", index=False) #Save Complement of Cases with no duplicates and no empty HLA
    CompCasesdrop = CompCases.drop_duplicates(subset = "Case", keep = 'first', inplace = False)
    RCompCasesdrop = RCompCases.drop_duplicates(subset = "Case", keep = 'first', inplace = False)
    #print(CompCasesdrop.shape)
    #print(Casesdrop.head())
    Casesdrop = Casesdrop.reset_index(drop=True)
    RCasesdrop = RCasesdrop.reset_index(drop=True)
    CompCasesdrop = CompCasesdrop.reset_index(drop=True)
    RCompCasesdrop = RCompCasesdrop.reset_index(drop=True)
    #CompCasesdrop.to_csv(path + save + "-Comp.csv", index=False) #Save Complement of Cases with no duplicates and no empty HLA
    
    Casesdropsurv = Casesdrop.merge(clinical,on='Case', how='left')
    RCasesdropsurv = RCasesdrop.merge(clinical,on='Case', how='left')
    #RCases.to_csv(path + save + "-Replicative.csv", index=False)
    CompCasesdropsurv = CompCasesdrop.merge(clinical,on='Case', how='left')
    RCompCasesdropsurv = RCompCasesdrop.merge(clinical,on='Case', how='left')
    #Casesdropsurv.to_csv(path + save + "-surv.csv", index=False)
    #CompCasesdropsurv.to_csv(path + save + "-Compsurv.csv", index=False) 
    results = logrank_test(Casesdropsurv['Days to Last Known Disease Status'], CompCasesdropsurv['Days to Last Known Disease Status'], Casesdropsurv['Death'], CompCasesdropsurv['Death'])
    #print(item, "  ", results.p_value)
    #results.print_summary() #details of each
    #print("P-value of " + item) #if you want all p-values
    #print(results.p_value) #if you want all p-values    
    if results.p_value < .05:
        #results.print_summary()
        #Casesdropsurv.to_csv(path + save + "-surv.csv", index=False)
        #CompCasesdropsurv.to_csv(path + save + "-Compsurv.csv", index=False) 
        datap = [[item, results.p_value, Casesdrop.shape[0], CompCasesdrop.shape[0]]]
        Rresults = logrank_test(RCasesdropsurv['Days to Last Known Disease Status'], RCompCasesdropsurv['Days to Last Known Disease Status'], RCasesdropsurv['Death'], RCompCasesdropsurv['Death'])
        Rdatap = [[item + "-Rep", Rresults.p_value, RCasesdrop.shape[0], RCompCasesdrop.shape[0]]]
        p = pd.DataFrame(datap, columns=['Combination','Log-rank Test p-value', 'Sample Size', 'Complement Sample Size'])
        Rp = pd.DataFrame(Rdatap, columns=['Combination','Log-rank Test p-value', 'Sample Size', 'Complement Sample Size'])
        ps = ps.append(p, ignore_index=True)
        ps = ps.append(Rp, ignore_index =True)
        psHLAVJ = psHLAVJ.append(p,ignore_index=True)
        psHLAVJ = psHLAVJ.append(Rp,ignore_index=True)
        #KMFit = KMF(Casesdropsurv, CompCasesdropsurv, item, save)
        #KMFit.KMFplot()
        #RKMFit = KMF(RCases, CompCasesdropsurv, item + "-Replicative", save + "-Replicative")
        #RKMFit.KMFplot()
print("HLA-VJ done")

#print significant ps    
print(ps[:]) 
ps.to_csv(path + "ps.csv")
#print(psHLAVJ[:])

#For significant combinations, find the p-values of arms
psHLAVJ = psHLAVJ[~psHLAVJ['Combination'].str.contains("-Rep")]
allpsHLA = allpsHLA.rename(columns={'Combination':'Combination-HLA', 'Log-rank Test p-value':'HLA Log-rank Test p-value', 'Sample Size':'HLA Sample Size', 'Complement Sample Size':'HLA Complement Sample Size'})  
allpsVJ = allpsVJ.rename(columns={'Combination':'Combination-VJ', 'Log-rank Test p-value':'VJ Log-rank Test p-value', 'Sample Size':'VJ Sample Size', 'Complement Sample Size':'VJ Complement Sample Size'})
stringpsHLAVJ = psHLAVJ['Combination'].str.split("-", expand=True) #splits into HLA type and V/J ID
stringpsHLAVJ.columns = ['Combination-HLA', 'Combination-VJ','2']
stringpsHLAVJ['Combination'] = psHLAVJ['Combination'] #obtaining HLA type from combo
stringpsHLAVJ['Combination-VJ'] = stringpsHLAVJ['Combination-VJ'].str.cat(stringpsHLAVJ['2'], sep = "-") 
stringpsHLAVJ['Combination-VJ'] = stringpsHLAVJ.apply(lambda x: x['Combination'].replace(x["Combination-HLA"], "").strip(), axis=1)
stringpsHLAVJ['Combination-VJ'] = stringpsHLAVJ['Combination-VJ'].str[1:]
#print(stringpsHLAVJ[:])
psHLAVJ['Combination-HLA'] = stringpsHLAVJ['Combination-HLA']
psHLAVJ['Combination-VJ'] = stringpsHLAVJ['Combination-VJ']
psHLAVJ = psHLAVJ.merge(allpsHLA,on='Combination-HLA', how = 'left') #using HLA strings, get relevant p-values
psHLAVJ = psHLAVJ.merge(allpsVJ,on='Combination-VJ', how = 'left') #using V/J ID strings, get relevant p-values
print(psHLAVJ[:])
psHLAVJ.to_csv(path + "pscombo,arms.csv")
#COX
pscox = ps[~ps['Combination'].str.contains("-Rep")]
pscox = pscox['Combination']
HLAdropcox = HLAdrop.merge(clinicalcox,on='Case', how = 'left')
TRBHLAVJcox = TRBHLAVJ.merge(clinicalcox, on = 'Case', how = 'left')
coxsummary = pd.DataFrame()
OS_cox_p_age = []
OS_cox_bcoef_age = []
OS_cox_p_gender = []
OS_cox_bcoef_gender = []
OS_cox_p_stage = []
OS_cox_bcoef_stage = []
for index, item in pscox.items():
    print(item)
    HLAtype = item.split('*')[0]
    HLAtype = "HLA-" + HLAtype #HLA-DPB
    VJonly = False
    HLAonly = False
    if "V" in item and ':' not in item: 
        column1 = 'VID'
        VJonly = True
    elif "J" in item and ':' not in item:
        column1 = 'JID'
        VJonly = True
    elif "V" in item: 
        column1 = HLAtype + ' + V-type'
        column2 = HLAtype + '\' + V-type'
    elif "J" in item:
        column1 = HLAtype + ' + J-type'
        column2 = HLAtype + '\' + J-type'
    else:
        column1 = HLAtype
        column2 = HLAtype + '\''
        HLAonly = True 

    if HLAonly is True:
        Used = HLAdropcox
    else:
        Used = TRBHLAVJcox
    is_c1 = Used[column1]==item
    if VJonly is True:
        is_c2 = False
    is_c2 = Used[column2]==item
    Cases = Used[is_c1|is_c2] # is item found in Column 1 or 2 (TRB-J/HLA-A or TRB-J/HLA-A')
    Casesdrop = Cases.drop_duplicates(subset = "Case", keep = 'first', inplace = False)
    Casesdrop = Casesdrop[['Case','Age','Gender','Stage','Days to Last Known Disease Status', 'Death']].copy()
    save = item.translate(str.maketrans('', '', '*:\'\/'))
    #Casesdrop.to_csv(path + save + "-COX.csv") you probably will want to keep clinicalcox from droppingna, put in 'Case' column 2 lines above this one 
    Casesdrop = Casesdrop.dropna() #Drops people who were missing a value of the above - potentially can bias people who don't have stages for their cancer bc they died
    cox = CoxPHFitter()
    try:
        cox.fit(Casesdrop,duration_col='Days to Last Known Disease Status',event_col='Death')
        cox.print_summary(3,"ascii")
        OS_cox = cox.summary
        #print(OS_cox.head())
        #OS_cox.to_csv(path + "OS.csv")
        OS_cox_p_age.append(OS_cox.iloc[0][8])
        OS_cox_bcoef_age.append(OS_cox.iloc[0][1])
        OS_cox_p_gender.append(OS_cox.iloc[1][8])
        OS_cox_bcoef_gender.append(OS_cox.iloc[1][1])
        OS_cox_p_stage.append(OS_cox.iloc[2][8])
        OS_cox_bcoef_stage.append(OS_cox.iloc[2][1])
    except:
        print(traceback.format_exc())
coxsummary["Combination"] = pscox
coxsummary["OS_cox_p_age"] = OS_cox_p_age
coxsummary["OS_cox_bcoef_age"] = OS_cox_bcoef_age
coxsummary["OS_cox_p_gender"] = OS_cox_p_gender
coxsummary["OS_cox_bcoef_gender"] = OS_cox_bcoef_gender
coxsummary["OS_cox_p_stage"] = OS_cox_p_stage
coxsummary["OS_cox_bcoef_stage"] = OS_cox_bcoef_stage
#coxsummary.to_csv(path + "coxsummary.csv")