@Library('blossom-github-lib@master') 
import ipp.blossom.*

podTemplate(cloud:'blsm-prod-cloud', yaml : """
  apiVersion: v1
  kind: Pod
  spec:
    serviceAccountName: "bionemo-nvmolkit-sa"
    nodeSelector:
      kubernetes.io/os: linux""",
  containers: [
    containerTemplate(name: 'cuda12', image: 'nvcr.io/nvidia/cuda:12.6.3-devel-ubuntu22.04', ttyEnabled: true, command: 'cat'),
    containerTemplate(name: 'cuda13', image: 'nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04', ttyEnabled: true, command: 'cat'),
  ]) {
      node(POD_LABEL) {
          def githubHelper
          stage('Get Token') {
              withCredentials([usernamePassword(credentialsId: 'github-token', passwordVariable: 'GIT_PASSWORD', usernameVariable: 'GIT_USERNAME')]) {
                  // create new instance of helper object
                  githubHelper = GithubHelper.getInstance("${GIT_PASSWORD}", githubData)
                  
              }
              
          }
          def stageName = '' 
          try {
              currentBuild.description = githubHelper.getBuildDescription()
              stageName = 'Code checkout'
              stage(stageName) {
                  // update status on github
                  githubHelper.updateCommitStatus("$BUILD_URL", "$stageName Running", GitHubCommitState.PENDING)
                  if("Open".equalsIgnoreCase(githubHelper.getPRState())){
                    println "PR State is Open"
                    // checkout head of pull request
                    checkout changelog: true, poll: true, scm: [$class: 'GitSCM', branches: [[name: "pr/"+githubHelper.getPRNumber()]],                   doGenerateSubmoduleConfigurations: false,
                    submoduleCfg: [],
                    userRemoteConfigs: [[credentialsId: 'github-token', url: githubHelper.getCloneUrl(), refspec: '+refs/pull/*/head:refs/remotes/origin/pr/*']]]
                  } 
                  else if("Merged".equalsIgnoreCase(githubHelper.getPRState())){
                    println "PR State is Merged"
                    // use following if you want to build merged code of the head & base branch
                    // ref : https://developer.github.com/v3/pulls/
                    checkout changelog: true, poll: true, scm: [$class: 'GitSCM', branches: [[name: githubHelper.getMergedSHA()]],
                    doGenerateSubmoduleConfigurations: false,
                    submoduleCfg: [],
                    userRemoteConfigs: [[credentialsId: 'github-token', url: githubHelper.getCloneUrl(), refspec: '+refs/pull/*/merge:refs/remotes/origin/pr/*']]]
                  }
              }
          
             stageName = 'Parallel Testing'
             stage(stageName) {
                 parallel(
                   'build_test_oldest_supported': {
                       container('cuda12') {
                           try {
                               githubHelper.updateCommitStatus("$BUILD_URL", "build_test_oldest_supported Running", GitHubCommitState.PENDING)
                               println "Setting up conda environment: Python 3.10, RDKit 2024.9.6"
                               sh 'cd ${WORKSPACE} && bash admin/setup_conda_env.sh 3.10 2024.9.6'
                               println "Running build and test on oldest supported CUDA"
                               // Build and test logic for CUDA 12.6 will be added here
                              sh 'source /usr/local/anaconda/etc/profile.d/conda.sh && conda activate base && mkdir /build && cd /build && cmake ${WORKSPACE} && make -j && ctest'
                               
                               githubHelper.updateCommitStatus("$BUILD_URL", "build_test_oldest_supported Complete", GitHubCommitState.SUCCESS)
                           } catch (Exception e) {
                               githubHelper.updateCommitStatus("$BUILD_URL", "build_test_oldest_supported Failed", GitHubCommitState.FAILURE)
                               throw e
                           }
                       }
                   },
                   'build_test_newest_supported': {
                       container('cuda13') {
                           try {
                               githubHelper.updateCommitStatus("$BUILD_URL", "build_test_newest_supported Running", GitHubCommitState.PENDING)
                               println "Setting up conda environment: Python 3.13, RDKit 2025.3.1"
                               sh 'cd ${WORKSPACE} && bash admin/setup_conda_env.sh 3.13 2025.3.1'
                               println "Running build and test on newest supported CUDA"
                               // Build and test logic for CUDA 13.0 will be added here
                              sh 'source /usr/local/anaconda/etc/profile.d/conda.sh && conda activate base && mkdir /build && cd /build && cmake ${WORKSPACE} && make -j && ctest'

                               githubHelper.updateCommitStatus("$BUILD_URL", "build_test_newest_supported Complete", GitHubCommitState.SUCCESS)
                           } catch (Exception e) {
                               githubHelper.updateCommitStatus("$BUILD_URL", "build_test_newest_supported Failed", GitHubCommitState.FAILURE)
                               throw e
                           }
                       }
                   },
                  'QA_pipeline': {
                      container('cuda13') {
                          try {
                              githubHelper.updateCommitStatus("$BUILD_URL", "QA_pipeline Running", GitHubCommitState.PENDING)
                              println "Setting up conda environment: Python 3.13, RDKit 2025.3.1"
                              sh 'cd ${WORKSPACE} && bash admin/setup_conda_env.sh 3.13 2025.3.1'
                              println "Running QA pipeline"
                              //sh 'cd ${WORKSPACE} && bash admin/run_qa.sh'

                              githubHelper.updateCommitStatus("$BUILD_URL", "QA_pipeline Complete", GitHubCommitState.SUCCESS)
                          } catch (Exception e) {
                              githubHelper.updateCommitStatus("$BUILD_URL", "QA_pipeline Failed", GitHubCommitState.FAILURE)
                              throw e
                          }
                      }
                  }
                 )
             }
              // upload jenkins job logs to github for external users
              // this function remove sensitive data from logs before upload
              // user can provide list of plain guard words (sensitive words) or regular expression for guard words 
              // def guardWords = ["gitlab-master.nvidia.com"]
              // 
              // githubHelper.uploadLogs(this, env.JOB_NAME, env.BUILD_NUMBER, guardWords, <extraGuardWordRegEx>) 
              githubHelper.uploadLogs(this, env.JOB_NAME, env.BUILD_NUMBER, null, null)

              // update status on github
              githubHelper.updateCommitStatus("$BUILD_URL", "Complete", GitHubCommitState.SUCCESS)
          }
          catch (Exception ex){
              currentBuild.result = 'FAILURE'
              println ex
              // upload jenkins job logs to github for external users
              // this function remove sensitive data from logs before upload
              // user can provide list of plain guard words (sensitive words) or regular expression for guard words 
              // def guardWords = ["gitlab-master.nvidia.com"]
              // 
              // githubHelper.uploadLogs(this, env.JOB_NAME, env.BUILD_NUMBER, guardWords, <extraGuardWordRegEx>) 
              githubHelper.uploadLogs(this, env.JOB_NAME, env.BUILD_NUMBER, null, null) 
              githubHelper.updateCommitStatus("$BUILD_URL", "$stageName Failed", GitHubCommitState.FAILURE)
          }
          
      }
      
  }
