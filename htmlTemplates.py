css = '''
<style>  
.chat-message {  
    padding: 1.5rem;   
    border-radius: 0.5rem;   
    margin-bottom: 1rem;   
    display: flex;  
}  
.chat-message.user {  
    background-color: #90caf9; /* Light Blue for User messages */  
    color: #000; /* Black text color for better contrast */  
    border-radius: 10px;  
    padding: 10px;  
    margin: 5px 0;  
}  

.chat-message.bot {  
    background-color: #ffe57f; /* Soft Yellow for Bot messages */  
    color: #000; /* Black text color for better contrast */  
    border-radius: 10px;  
    padding: 10px;  
    margin: 5px 0;  
}  
.chat-message .avatar {  
    width: 20%;  
}  
.chat-message .avatar img {  
    max-width: 78px;  
    max-height: 78px;  
    border-radius: 50%;  
    object-fit: cover;  
}  
.chat-message .message {  
    width: 80%;  
    padding: 0 1.5rem;  
    color: #000; /* Change to black for better readability */  
}  
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.postimg.cc/JD5VLWFt/assisstant.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.postimg.cc/62s1q1ZR/user.webp" alt="user" border="0">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
