import { initializeApp } from "https://www.gstatic.com/firebasejs/12.3.0/firebase-app.js";
 import {getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword, signOut} from "https://www.gstatic.com/firebasejs/12.3.0/firebase-auth.js"
 import {getFirestore, setDoc, doc} from "https://www.gstatic.com/firebasejs/12.3.0/firebase-firestore.js" 

   
  const firebaseConfig = {
    apiKey: "AIzaSyDtrNwHLP5MwGwkd8lk6GyrxkecDFd4G0Q",
    authDomain: "pet-care-7921f.firebaseapp.com",
    databaseURL: "https://pet-care-7921f-default-rtdb.firebaseio.com",
    projectId: "pet-care-7921f",
    storageBucket: "pet-care-7921f.firebasestorage.app",
    messagingSenderId: "1031801172442",
    appId: "1:1031801172442:web:49e72393b702de161890e3"
  };

  
  const app = initializeApp(firebaseConfig);
 

   function showMessage(message, divId){
    var messageDiv=document.getElementById(divId);
    messageDiv.style.display="block";
    messageDiv.innerHTML=message;
    messageDiv.style.opacity=1;
    setTimeout(function(){
        messageDiv.style.opacity=0;
    },5000);
 }
 const signUp=document.getElementById('submitSignUp');
 signUp.addEventListener('click', (event)=>{
    event.preventDefault();
    const email=document.getElementById('rEmail').value;
    const password=document.getElementById('rPassword').value;
    const firstName=document.getElementById('fName').value;
    const lastName=document.getElementById('lName').value;

    const auth=getAuth();
    const db=getFirestore();

    createUserWithEmailAndPassword(auth, email, password)
    .then((userCredential)=>{
        const user=userCredential.user;
        const userData={
            email: email,
            firstName: firstName,
            lastName:lastName
        };
        showMessage('Account Created Successfully', 'signUpMessage');
        const docRef=doc(db, "users", user.uid);
        setDoc(docRef,userData)
        .then(()=>{
            window.location.href='index.html';
        })
        .catch((error)=>{
            console.error("error writing document", error);

        });
    })
    .catch((error)=>{
        const errorCode=error.code;
        if(errorCode=='auth/email-already-in-use'){
            showMessage('Email Address Already Exists !!!', 'signUpMessage');
        }
        else{
            showMessage('unable to create User', 'signUpMessage');
        }
    })
 });

 const signIn=document.getElementById('submitSignIn');
 signIn.addEventListener('click', (event)=>{
    event.preventDefault();
    const email=document.getElementById('email').value;
    const password=document.getElementById('password').value;
    const auth=getAuth();

    signInWithEmailAndPassword(auth, email,password)
    .then((userCredential)=>{
        showMessage('login is successful', 'signInMessage');
        const user=userCredential.user;
        localStorage.setItem('loggedInUserId', user.uid);
        window.location.href='index.html'; // Redirect fixed to index.html
    })
    .catch((error)=>{
        const errorCode=error.code;
        if(errorCode==='auth/invalid-credential'){
            showMessage('Incorrect Email or Password', 'signInMessage');
        }
        else{
            showMessage('Account does not Exist', 'signInMessage');
        }
    })
 });

// --- NEW LOGOUT AND STATUS FUNCTIONS ---

/**
 * Checks if a user is currently logged in based on the localStorage flag.
 * @returns {boolean} True if a user ID is found in localStorage.
 */
export function checkLoginStatus() {
    return localStorage.getItem('loggedInUserId') !== null;
}

/**
 * Handles the Firebase Sign Out process, clears local storage, and redirects.
 */
export function handleLogout() {
    const auth = getAuth();
    signOut(auth).then(() => {
        // Sign-out successful. Clear the localStorage item and redirect to the home page.
        localStorage.removeItem('loggedInUserId');
        window.location.href = 'index.html';
    }).catch((error) => {
        console.error("Logout failed:", error);
        alert("Logout failed. Please try again.");
    });
}