// Ttt
#include <cstdlib>

#include <stdio.h>
#include <cuda_runtime.h>

#include <iostream>

#include <math.h>
#include <chrono>
#include <ctime>

#include "../src/graphcode.h"
#include "../src/cudahelper.cuh"
#include "../src/helper.h"


std::vector<std::string> DICT = {"root-asset","date","null","type","blog","author","washington post staff","source","the washington post","title","benghazi","libya","transcript","skip","quorum","alert","witness","deserve","decorum","disrupt","proper","incremental","ranking","recognized","stevens","wood","courage","gate","machine","mortar","pour","sacrifice","everlasting","gratitude","owe","anniversary","9/11","compound","pursuing","narrow","scope","linger","incapable","unwilling","relevant","accounting","mail","knowledgeable","server","custody","uncover","testimony","particularity","accountability","thorough","ambassador","frank","soil","fundamental","obligation","purport","hundred","arrangement","exclusive","sole","amendment","privilege","incrimination","annals","experiment","examine","pledged","manner","worthy","memory","pursue","definitive","profile","brave","bipartisan","respected","caucus","deadline","unlimited","squandering","taxpayer","derail","reasonable","abusive","credibility","mccarthy","speaker","zero","cross","quote","justified","jurisdictional","crossed","dying","graham","hell","wild","objection","database","forth","spin","conspiracy","accepting","inaccurate","sidney","checker","rating","deposit","fishing","expedition","corps","diplomatic","yield","gentleman","admired","diplomat","accomplished","sand","shoe","liked","revolution","envoy","gathering","libyan","dictator","greek","cargo","ship","diplomacy","testament","awareness","ten","islam","prophet","personally","valued","pretoria","baghdad","montreal","outpost","distance","afghanistan","trained","distinguished","globe","haiti","tenure","painful","marines","casket","plane","andrews","tragedy","prior","hard-hitting","boot","operate","understood","terrorism","achieve","inevitably","contractor","locally","employed","bunker","devoted","root","aggressor","vacuum","everywhere","egypt","saudi","jerusalem","intifada","extremist","instability","rampant","pivotal","unrest","democracy","eastern","fragile","foothold","tunisia","destabilize","airplane","sky","israeli","airport","nearest","dignity","anywhere","substitute","embassy","resort","creative","profession","dedicated","observation","safely","branch","deadly","reagan","marine","barrack","bomb","kenya","tanzania","longest","appointed","institution","boards","findings","africa","punch","implementation","slate","deployment","deploy","subsequent","joint","pride","ideology","abroad","cooperation","landmark","treaty","myanmar","democratic","patriotism","loyalty","resist","disagree","partisan","bestow","statesmanship","shrift","precis","appreciated","illinois","sullivan","memo","tripoli","rebel","tightening","noose","instrumental","pause","considerable","boil","arab","rose","chart","demanding","genocide","hunt","cockroach","intensity","shore","strongly","idly","unintended","diligence","heading","enormous","dig","casualty","remarkable","intervention","mull","tactical","courtesy","obstacle","gates","advocate","urging","in-person","assumed","resolution","unprecedented","instruction","capacity","bulk","militarily","convinced","veto","persuade","abstain","congressman","joining","legitimacy","behest","varying","bin","intense","inaudible","humanitarian","launching","strategic","totality","overcome","drove","load","vital","recall","monetary","sum","interchange","architect","disaster","yesterday","unanswered","implication","rightly","handled","nonpartisan","importantly","judgment","conducting","lamb","supervisor","cable","signature","falsely","clip","outright","whopper","copy","stamp","compile","charleston","detailed","confusion","clarify","tradition","actionable","accident","historically","scrutiny","indiana","ms","brooks","drawing","pile","relate","hourly","handful","disparity","seated","conclude","representative","hand-pick","paris","reconnaissance","insurgency","fashioned","anticipated","assessment","malta","groundwork","lay","recollection","refresh","interrupt","expeditionary","undertaking","practiced","communication","instantaneous","undertake","anxious","depart","worsened","guide","overtly","constant","radar","explosive","device","behalf","classified","bet","impression","sharing","thank","informed","briefing","one-on-one","weekly","briefcase","occasions","recommendation","conjunction","questioning","procedure","handling","dedication","overseas","affinity","handshake","globally","safeguard","wherever","heroic","ideal","sufficient","contingency","communications","assets","assess","dempsey","guidance","kabul","incredibly","supportive","snapshot","counterpart","carter","useful","institutionalized","platoon","periodic","waste","latter","impress","institute","reward","litany","fence","lady","gentle","alabama","uncertainly","uncertainty","colleague","belong","noticed","desk","tab","encounter","lockdown","mrs","stacks","chemical","rid","proliferation","collected","destroyed","reducing","stocks","burns","please","visiting","numerous","consulate","vigorous","formal","drill","brush","frustrating","correspond","assigned","bass","acknowledge","civic","permission","requested","repel","unfortunate","annex","obsession","formed","hat","shame","suicide","embarrassing","pakistan","yemen","willingness","traveled","amazed","chafe","peshawar","vividly","driven","bias","ear","blind","guarded","journal","function","editorial","orderly","dysfunction","substance","stalk","arkansas","interpreter","bunch","militant","protected","referenced","spoken","publication","stability","equipped","mine","practically","rotated","investigate","supplement","fortification","unfortunately","kansas","summary","paycheck","constantly","connected","breach","duty","legally","absence","dereliction","channels","occasionally","slope","perspective","poster","fulfilled","namely","flag","allegedly","wittingly","unwittingly","sanchez","allegation","volume","emphasis","hysterical","occasion","observe","jacob","queue","mitchell","logic","falsehood","correctly","description","factual","factually","incorrect","theory","aftermath","myth","debunk","rand","documentation","counter","shortly","defending","whirl","evacuate","distressing","congress","indicate","directed","deck","nowhere","cairo","victoria","spokesperson","spray","demonstration","eyewitness","rice","consequence","spontaneous","false","reservation","narrative","roof","grave","egyptian","inflammatory","attacker","justification","tunis","thankfully","appreciate","pressed","approximately","fluid","affiliated","conflicting","slide","rhodes","assume","alive","square","chapter","disservice","glad","confusing","motivated","khartoum","burning","tunisian","desperate","insinuation","plain","vastly","monitor","sake","scheme","entitled","premium","documentary","buck","exhibit","ceiling","recitation","sheer","metric","regularity","amazing","centric","aired","cancel","conclusion","investigator","ignore","damn","singly","briefly","merit","confused","accusation","sleep","wrack","brain","predecessor","nairobi","alike","east","optimist","prosecution","stack","sad","regardless","undisputed","bubble","solicit","redact","identifier","insight","keeper","tomorrow","intervene","electricity","definition","evaluate","print","determination","somewhere","thread","prolific","explanation","tyler","rhetoric","derision","seizing","jaw","unenthusiastic","paralysis","lukewarm","operational","pseudo","venture","madam","drivel","eminently","relevance","classification","subpoena","deposition","compelling","unilaterally","marshal","sir","morse","carrier","pigeon","smoke","irrelevant","telephone","bothered","motion","parliamentarian","mills","fairness","discovery","odd","outstanding","admission","selectively","indulgence","signify","clerk","adequately","soft","mailing","react","favorable","avenue","best","newly","lever","register","follower","incoming","adamantly","convey","engage","messaging","folder","medicine","gasoline","diesel","milk","visa","basically","affirmatively","positively","fuel","insult","servant","similarly","elements","unofficial","originally","buttonhole","reception","helpful","carefully","incredulity","testify","aisle","overwhelm","somehow","posed","historical","influential","deference","consent","reopen","full-time","sentiment","inhabit","permanent","promote","vibrant","bilateral","outweigh","elimination","profoundly","lacking","minimal","nonexistent","configuration","geography","mid-july","formally","hiding","headquarters","accurate","duration","dispute","predominantly","located","informative","update","gradual","deem","consul","protocol","accurately","disclosed","assignment","militia","disarm","miss","dated","universe","curious","humor","typing","picking","spirit","entrepreneurial","barrier","length","describe","patrick","high-level","evaluation","advised","abandon","contrary","unaware","prioritize","farthest","unquestionably","viewer","embarrassment","reference","prosecute","propel","passionate","rip","remove","juxtapose","difficulty","drag","atmosphere","depth","specify","undergo","circle","aunt","remainder","haven","inflict","inhalation","enable","endeavor","interior","proceeding","instinct","desperately","saving","succumb","horror","resuscitate","labored","fog","heroism","professionalism","refuge","fortified","whisper","fighter","god","one-tenth","row","forbid","soliloquy","elegant","finger","tape","rewind","regret","violation","cell","24/7","accordance","respects","breakfast","trick","follow-up","inadequate","grossly","magnitude","preventable","premises","vulnerable","lease","consensus","west","professionalize","height","province","requirement","dip","dialogue","boring","pedigree","honorable","malign","sic","retreated","accountable","oversight","mismanagement","price","inflexible","limb","armor","static","kinetic","flexibility","vet","reliable","invite","friendly","accompany","landed","firearm","vienna","trigger","funded","responsiveness","echo","expression","havoc","causing","therefore","explicit","berlin","rome","istanbul","appoint","laying","foundation","timeline","bother","key","explicitly","comprehensive","excerpt","cabinet","instruct","recognition","vacation","handed","guarantee","driveway","timing","suggestion","rescue","import","insofar","doctrine","publicize","coalition","sanction","imposed","disposal","extent","lap","precise","recess","mess","bolster","approve","approving","premise","distinction","listening","chaos","spike","cross","forthcoming","monitoring","consideration","departure","ultimate","oversee","properly","fault","mislead","pleased","beef","maximum","informing","tourist","criterion","extreme","conditioning","orientation","imperative","gavel","quicker","attached","run-up","append","hampshire","asset","spinning","suspicion","leon","undeniable","intentionally","pale","surveillance","drone","ordering","rota","spain","croatia","mobilize","redirect","bound","dispatch","satisfied","dissipated","actively","assertion","scrambled","logistics","upheaval","volatility","amateur","mock","afghan","tire","indonesian","jakarta","volatile","prayer","battering","army","fancy"};
std::vector<std::string> DICT2 = {"root-asset","date","null","type","blog","author","washington post staff","source","the washington post","title","benghazi","libya","transcript","skip","quorum","alert","witness","deserve","decorum","disrupt","proper","incremental","ranking","recognized","stevens","wood","courage","gate","machine","mortar","pour","sacrifice","everlasting","gratitude","owe","anniversary","9/11","compound","pursuing","narrow","scope","linger","incapable","unwilling","relevant","accounting","mail","knowledgeable","server","custody","uncover","testimony","particularity","accountability","thorough","ambassador","frank","soil","fundamental","obligation","purport","hundred","arrangement","exclusive","sole","amendment","privilege","incrimination","annals","experiment","examine","pledged","manner","worthy","memory","pursue","definitive","profile","brave","bipartisan","respected","caucus","deadline","unlimited","squandering","taxpayer","derail","reasonable","abusive","credibility","mccarthy","speaker","zero","cross","quote","justified","jurisdictional","crossed","dying","graham","hell","wild","objection","database","forth","spin","conspiracy","accepting","inaccurate","sidney","checker","rating","deposit","fishing","expedition","corps","diplomatic","yield","gentleman","admired","diplomat","accomplished","sand","shoe","liked","revolution","envoy","gathering","libyan","dictator","greek","cargo","ship","diplomacy","testament","awareness","ten","islam","prophet","personally","valued","pretoria","baghdad","montreal","outpost","distance","afghanistan","trained","distinguished","globe","haiti","tenure","painful","marines","casket","plane","andrews","tragedy","prior","hard-hitting","boot","operate","understood","terrorism","achieve","inevitably","contractor","locally","employed","bunker","devoted","root","aggressor","vacuum","everywhere","egypt","saudi","jerusalem","intifada","extremist","instability","rampant","pivotal","unrest","democracy","eastern","fragile","foothold","tunisia","destabilize","airplane","sky","israeli","airport","nearest","dignity","anywhere","substitute","embassy","resort","creative","profession","dedicated","observation","safely","branch","deadly","reagan","marine","barrack","bomb","kenya","tanzania","longest","appointed","institution","boards","findings","africa","punch","implementation","slate","deployment","deploy","subsequent","joint","pride","ideology","abroad","cooperation","landmark","treaty","myanmar","democratic","patriotism","loyalty","resist","disagree","partisan","bestow","statesmanship","shrift","precis","appreciated","illinois","sullivan","memo","tripoli","rebel","tightening","noose","instrumental","pause","considerable","boil","arab","rose","chart","demanding","genocide","hunt","cockroach","intensity","shore","strongly","idly","unintended","diligence","heading","enormous","dig","casualty","remarkable","intervention","mull","tactical","courtesy","obstacle","gates","advocate","urging","in-person","assumed","resolution","unprecedented","instruction","capacity","bulk","militarily","convinced","veto","persuade","abstain","congressman","joining","legitimacy","behest","varying","bin","intense","inaudible","humanitarian","launching","strategic","totality","overcome","drove","load","vital","recall","monetary","sum","interchange","architect","disaster","yesterday","unanswered","implication","rightly","handled","nonpartisan","importantly","judgment","conducting","lamb","supervisor","cable","signature","falsely","clip","outright","whopper","copy","stamp","compile","charleston","detailed","confusion","clarify","tradition","actionable","accident","historically","scrutiny","indiana","ms","brooks","drawing","pile","relate","hourly","handful","disparity","seated","conclude","representative","hand-pick","paris","reconnaissance","insurgency","fashioned","anticipated","assessment","malta","groundwork","lay","recollection","refresh","interrupt","expeditionary","undertaking","practiced","communication","instantaneous","undertake","anxious","depart","worsened","guide","overtly","constant","radar","explosive","device","behalf","classified","bet","impression","sharing","thank","informed","briefing","one-on-one","weekly","briefcase","occasions","recommendation","conjunction","questioning","procedure","handling","dedication","overseas","affinity","handshake","globally","safeguard","wherever","heroic","ideal","sufficient","contingency","communications","assets","assess","dempsey","guidance","kabul","incredibly","supportive","snapshot","counterpart","carter","useful","institutionalized","platoon","periodic","waste","latter","impress","institute","reward","litany","fence","lady","gentle","alabama","uncertainly","uncertainty","colleague","belong","noticed","desk","tab","encounter","lockdown","mrs","stacks","chemical","rid","proliferation","collected","destroyed","reducing","stocks","burns","please","visiting","numerous","consulate","vigorous","formal","drill","brush","frustrating","correspond","assigned","bass","acknowledge","civic","permission","requested","repel","unfortunate","annex","obsession","formed","hat","shame","suicide","embarrassing","pakistan","yemen","willingness","traveled","amazed","chafe","peshawar","vividly","driven","bias","ear","blind","guarded","journal","function","editorial","orderly","dysfunction","substance","stalk","arkansas","interpreter","bunch","militant","protected","referenced","spoken","publication","stability","equipped","mine","practically","rotated","investigate","supplement","fortification","unfortunately","kansas","summary","paycheck","constantly","connected","breach","duty","legally","absence","dereliction","channels","occasionally","slope","perspective","poster","fulfilled","namely","flag","allegedly","wittingly","unwittingly","sanchez","allegation","volume","emphasis","hysterical","occasion","observe","jacob","queue","mitchell","logic","falsehood","correctly","description","factual","factually","incorrect","theory","aftermath","myth","debunk","rand","documentation","counter","shortly","defending","whirl","evacuate","distressing","congress","indicate","directed","deck","nowhere","cairo","victoria","spokesperson","spray","demonstration","eyewitness","rice","consequence","spontaneous","false","reservation","narrative","roof","grave","egyptian","inflammatory","attacker","justification","tunis","thankfully","appreciate","pressed","approximately","fluid","affiliated","conflicting","slide","rhodes","assume","alive","square","chapter","disservice","glad","confusing","motivated","khartoum","burning","tunisian","desperate","insinuation","plain","vastly","monitor","sake","scheme","entitled","premium","documentary","buck","exhibit","ceiling","recitation","sheer","metric","regularity","amazing","centric","aired","cancel","conclusion","investigator","ignore","damn","singly","briefly","merit","confused","accusation","sleep","wrack","brain","predecessor","nairobi","alike","east","optimist","prosecution","stack","sad","regardless","undisputed","bubble","solicit","redact","identifier","insight","keeper","tomorrow","intervene","electricity","definition","evaluate","print","determination","somewhere","thread","prolific","explanation","tyler","rhetoric","derision","seizing","jaw","unenthusiastic","paralysis","lukewarm","operational","pseudo","venture","madam","drivel","eminently","relevance","classification","subpoena","deposition","compelling","unilaterally","marshal","sir","morse","carrier","pigeon","smoke","irrelevant","telephone","bothered","motion","parliamentarian","mills","fairness","discovery","odd","outstanding","admission","selectively","indulgence","signify","clerk","adequately","soft","mailing","react","favorable","avenue","best","newly","lever","register","follower","incoming","adamantly","convey","engage","messaging","folder","medicine","gasoline","diesel","milk","visa","basically","affirmatively","positively","fuel","insult","servant","similarly","elements","unofficial","originally","buttonhole","reception","helpful","carefully","incredulity","testify","aisle","overwhelm","somehow","posed","historical","influential","deference","consent","reopen","full-time","sentiment","inhabit","permanent","promote","vibrant","bilateral","outweigh","elimination","profoundly","lacking","minimal","nonexistent","configuration","geography","mid-july","formally","hiding","headquarters","accurate","duration","dispute","predominantly","located","informative","update","gradual","deem","consul","protocol","accurately","disclosed","assignment","militia","disarm","miss","dated","universe","curious","humor","typing","picking","spirit","entrepreneurial","barrier","length","describe","patrick","high-level","evaluation","advised","abandon","contrary","unaware","prioritize","farthest","unquestionably","viewer","embarrassment","reference","prosecute","propel","passionate","rip","remove","juxtapose","difficulty","drag","atmosphere","depth","specify","undergo","circle","aunt","remainder","haven","inflict","inhalation","enable","endeavor","interior","proceeding","instinct","desperately","saving","succumb","horror","resuscitate","labored","fog","heroism","professionalism","refuge","fortified","whisper","fighter","god","one-tenth","row","forbid","soliloquy","elegant","finger","tape","rewind","regret","violation","cell","24/7","accordance","respects","breakfast","trick","follow-up","inadequate","grossly","magnitude","preventable","premises","vulnerable","lease","consensus","west","professionalize","height","province","requirement","dip","dialogue","boring","pedigree","honorable","malign","sic","retreated","accountable","oversight","mismanagement","price","inflexible","limb","armor","static","kinetic","flexibility","vet","reliable","invite","friendly","accompany","landed","firearm","vienna","trigger","funded","responsiveness","echo","expression","havoc","causing","therefore","explicit","berlin","rome","istanbul","appoint","laying","foundation","timeline","bother","key","explicitly","comprehensive","excerpt","cabinet","instruct","recognition","vacation","handed","guarantee","driveway","timing","suggestion","rescue","import","insofar","doctrine","publicize","coalition","sanction","imposed","disposal","extent","lap","precise","recess","mess","bolster","approve","approving","premise","distinction","listening","chaos","spike","cross","forthcoming","monitoring","consideration","departure","ultimate","oversee","properly","fault","mislead","pleased","beef","maximum","informing","tourist","criterion","extreme","conditioning","orientation","imperative","gavel","quicker","attached","run-up","append","hampshire","asset","spinning","suspicion","leon","undeniable","intentionally","pale","surveillance","drone","ordering","rota","spain","croatia","mobilize","redirect","bound","dispatch","satisfied","dissipated","actively","assertion","scrambled","logistics","upheaval","volatility","amateur","mock","afghan","tire","indonesian","jakarta","volatile","prayer","battering","army","fancy"};
#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 16

#define Nrows 4
#define Ncols 4

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int hostPtr, int b){ return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); }

json generateTestData(int n);

void testFindLargestDivisor();

int findLargestDivisor(int n);

bool isPrime(int number);

__global__ void calcMetrices(int *data, int *comparedata, unsigned long matrixSize, int *pInt, int *pInt1) {

    //int tid = blockIdx.x;
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int /*offset*/ tid = x + y * blockDim.x * gridDim.x;

    //if (tid > matrixSize-10000) return;

     int q = sqrt((float)matrixSize);

     for (int i = 0; i < q; i++) {
        if (tid == i*q+i) {
            //Can be used to debug
            //pInt[tid] = -1;
            return;
        }
     }

    if (data[tid] != 0 ) {
        if (comparedata[tid] != 0) {
            pInt1[tid] = 1;
            if (data[tid] == comparedata[tid]) {
                pInt[tid] = 1;
            }

        }
    }

}



/******************/
/* TEST KERNEL 2D */
/******************/

__global__ void test_kernel_2D(float *devPtr, size_t pitch)
{
    int    tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int    tidy = blockIdx.y*blockDim.y + threadIdx.y;

    if ((tidx < Ncols) && (tidy < Nrows))
    {
        float *row_a = (float *)((char*)devPtr + tidy * pitch);
        if (tidx == tidy) {
            row_a[tidx] = 0.0;
        } else {

            row_a[tidx] = row_a[tidx] * tidx * tidy;
        }
    }
}



/********/
/* MAIN */
/********/
int testCudaMatrixMemory()
{
    float hostPtr[Nrows][Ncols];
    float *devPtr;
    size_t pitch;

    for (int i = 0; i < Nrows; i++)
        for (int j = 0; j < Ncols; j++) {
            hostPtr[i][j] = 1.f;
            //printf("row %i column %i value %f \n", i, j, hostPtr[i][j]);
        }

    // --- 2D pitched allocation and host->device memcopy
    HANDLE_ERROR(cudaMallocPitch(&devPtr, &pitch, Ncols * sizeof(float), Nrows));
    HANDLE_ERROR(cudaMemcpy2D(devPtr, pitch, hostPtr, Ncols*sizeof(float), Ncols*sizeof(float), Nrows, cudaMemcpyHostToDevice));

    dim3 gridSize(iDivUp(Ncols, BLOCKSIZE_x), iDivUp(Nrows, BLOCKSIZE_y));
    dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);

    test_kernel_2D<<<gridSize, blockSize>>>(devPtr, pitch);

    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy2D(hostPtr, Ncols * sizeof(float), devPtr, pitch, Ncols * sizeof(float), Nrows, cudaMemcpyDeviceToHost));

    for (int i = 0; i < Nrows; i++)
        for (int j = 0; j < Ncols; j++)
            printf("row %i column %i value %f \n", i, j, hostPtr[i][j]);

    return 0;
}

void testCudaLinearMatrixMemory(){

    // Generate test data
    json gcq = generateTestData(2040);


    // Transform to data structures for calculations
    json gc1Dictionary = gcq["dictionary"];
    int numberOfElements = gc1Dictionary.size();

    int matrix1[gc1Dictionary.size()][gc1Dictionary.size()];
    convertDict2Matrix(gc1Dictionary.size(), (int *) matrix1, gcq["matrix"]);

    int items = numberOfElements * numberOfElements;
    std::cout << "Items: " << items << std::endl;
    int inputMatrix[items];

    int count = 0;
    for (int i = 0; i < numberOfElements; i++)
        for (int j = 0; j < numberOfElements; j++) {
            inputMatrix[count++] = matrix1[i][j];
        }

    // Prep for cuda


    int *gpu_inputMatrix;
    int *darr_edge_metric_count;
    int *darr_num_of_non_zero_edges;
    // Allocate device memory for inputMatrix
    //cudaMalloc((void**)&gpu_inputMatrix, sizeof(int) );

    auto start = std::chrono::system_clock::now();

    HANDLE_ERROR(cudaMalloc((void**)&gpu_inputMatrix, sizeof(int) * items) );
    HANDLE_ERROR(cudaMalloc((void**)&darr_edge_metric_count, sizeof(int) * items) );
    HANDLE_ERROR(cudaMalloc((void**)&darr_num_of_non_zero_edges, sizeof(int) * items) );
    /*
    cudaMemcpy2DToArray (dst,
                         0,
                         0,
                         matrix1,
                         sizeof(int),
                         gc1Dictionary.size() * sizeof(int),
                         gc1Dictionary.size(),
                         cudaMemcpyHostToDevice );

    */

    // Transfer data from host to device memory
    HANDLE_ERROR(cudaMemcpy(gpu_inputMatrix, inputMatrix, sizeof(int) * items, cudaMemcpyHostToDevice));

    //int gridSize = ceil(items / 1024.0);
    int gridSize = findLargestDivisor(items);



    int blockSize = findLargestDivisor(gridSize);
    if (blockSize == 0) {
        if (isPrime(gridSize)) {
            //blockSize= findLargestDivisor(gridSize+1);
            gridSize += 1;
            blockSize = findLargestDivisor(gridSize);
        }
    }
    dim3 block(blockSize, ceil(gridSize/(float) blockSize));
    dim3 grid(ceil(items/(float) gridSize));

    //HANDLE_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calcMetrices, 0, 0));

    // calculation
    auto loaded = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = loaded - start;

    std::cout << "elapsed time: " << elapsed_seconds.count() << ")"
              << std::endl;
    calcMetrices<<<grid, block>>>(gpu_inputMatrix, gpu_inputMatrix, items, darr_edge_metric_count,
                                   darr_num_of_non_zero_edges);



    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    auto end = std::chrono::system_clock::now();

    elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";
    // Retrieve results
    //int arr_edge_metric_count[items];
    int *arr_edge_metric_count = (int *) malloc(sizeof(int) * items);

    //int arr_num_of_non_zero_edges[items];
    int *arr_num_of_non_zero_edges = (int *) malloc(sizeof(int) * items);

    HANDLE_ERROR(cudaMemcpy(arr_edge_metric_count, darr_edge_metric_count, sizeof (int) * items, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(arr_num_of_non_zero_edges, darr_num_of_non_zero_edges, sizeof (int) * items, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(gpu_inputMatrix));
    HANDLE_ERROR(cudaFree(darr_edge_metric_count));
    HANDLE_ERROR(cudaFree(darr_num_of_non_zero_edges));

    // Result reduction
    int num_of_non_zero_edges = 0;
    int edge_metric_count = 0;
    for(int i = 0; i < items; i++) {
        //std::cout << "pos: " << i << " value: " << arr_edge_metric_count[i] << std::endl;
        if (arr_edge_metric_count[i] == 1) {
            edge_metric_count++;
        }
        if (arr_num_of_non_zero_edges[i] == 1) {
            num_of_non_zero_edges++;
        }
    }

    // Calculate metrices
    float node_metric = (float) numberOfElements / (float) gc1Dictionary.size();

    float edge_metric = 0.0;
    if (num_of_non_zero_edges > 0)
        edge_metric = (float) edge_metric_count / (float) num_of_non_zero_edges;


    std::cout << "Similarity: " << " value: " << node_metric << std::endl;
    std::cout << "Recommendation: " << " value: " << edge_metric << std::endl;

}

bool isPrime(int number) {
    bool isPrime = true;

    // 0 and 1 are not prime numbers
    if (number == 0 || number == 1) {
        isPrime = false;
    }
    else {
        for (int i = 2; i <= number / 2; ++i) {
            if (number % i == 0) {
                isPrime = false;
                break;
            }
        }

    }
    return isPrime;
}


json generateTestData(int n) {
    nlohmann::json gcq;
    //gcq["dictionary"] = { "head", "body"};
    int foo = DICT.size();

    DICT.insert( DICT.end(), DICT2.begin(), DICT2.end() );

    if (n > DICT.size()) {
        exit(71);
    }

    std::vector<std::string> newVec(DICT.begin(), DICT.begin() + n);
    //extractElements(DICT, subA, n);

    std::vector<std::vector<int>> data;
    for (int i = 0; i < n; i++) {
        std::vector<int> x;
        data.push_back(x);
        for (int j = 0; j < n; j++) {

            if (i == j ) {
                //data[i][j] = 1;
                data[i].push_back(1);
            } else {
                data[i].push_back(i+j%2);
            }
        }
    }

    gcq["dictionary"] = newVec;
    gcq["matrix"] = data;
    return gcq;
}



int main(int, char**)
{
    testFindLargestDivisor();
    testCudaMatrixMemory();
    testCudaLinearMatrixMemory();

}

void testFindLargestDivisor() {
    // Note that this loop runs till square root
    int d = findLargestDivisor(513);
    assert(d == 27);
    d = findLargestDivisor(73);
    assert(d == 0);

}

int findLargestDivisor(int n) {
    for (int i=sqrt(n); i < n; i++)
    {
        if (n%i == 0)
            return i;
    }
    return 0;
}
